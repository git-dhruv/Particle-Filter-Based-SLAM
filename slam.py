# Pratik Chaudhari (pratikac@seas.upenn.edu)

import os, sys, pickle, math
from copy import deepcopy

from scipy import io
import numpy as np
import matplotlib.pyplot as plt

from load_data import load_lidar_data, load_joint_data, joint_name_to_index
from utils import *
import logging

logger = logging.getLogger()
logger.setLevel(os.environ.get("LOGLEVEL", "INFO"))

class map_t:
    """
    This will maintain the occupancy grid and log_odds. You do not need to change anything
    in the initialization
    """
    def __init__(s, resolution=0.05):
        s.resolution = resolution
        s.xmin, s.xmax = -20, 20
        s.ymin, s.ymax = -20, 20
        s.szx = int(np.ceil((s.xmax-s.xmin)/s.resolution+1))
        s.szy = int(np.ceil((s.ymax-s.ymin)/s.resolution+1))

        # binarized map and log-odds
        s.cells = np.zeros((s.szx, s.szy), dtype=np.int8)
        s.log_odds = np.zeros(s.cells.shape, dtype=np.float64)

        # value above which we are not going to increase the log-odds
        # similarly we will not decrease log-odds of a cell below -max
        s.log_odds_max = 5e6
        # number of observations received yet for each cell
        s.num_obs_per_cell = np.zeros(s.cells.shape, dtype=np.uint64)

        # we call a cell occupied if the probability of
        # occupancy P(m_i | ... ) is >= occupied_prob_thresh
        s.occupied_prob_thresh = 0.6    #this is the way...
        s.log_odds_thresh = np.log(s.occupied_prob_thresh/(1-s.occupied_prob_thresh))

    def grid_cell_from_xy(s, x, y):
        """
        x and y are 1-dimensional arrays, compute the cell indices in the map corresponding
        to these (x,y) locations. You should return an array of shape 2 x len(x). Be
        careful to handle instances when x/y go outside the map bounds, you can use
        np.clip to handle these situations.
        """
        #### TODO: XXXXXXXXXXX
        # raise NotImplementedError
        xidx = (x-(s.xmin))/s.resolution
        yidx = (y-(s.ymin))/s.resolution 
        xidx = np.clip(np.abs(xidx),0,s.szx-1)
        yidx = np.clip(np.abs(yidx),0,s.szy-1)
        return np.array([xidx,yidx]).astype(int)

class slam_t:
    """
    s is the same as self. In Python it does not really matter
    what we call self, s is shorter. As a general comment, (I believe)
    you will have fewer bugs while writing scientific code if you
    use the same/similar variable names as those in the mathematical equations.
    """
    def __init__(s, resolution=0.05, Q=1e-3*np.eye(3),
                 resampling_threshold=0.3):
        #Init sensor mode, dynamic noise and map
        s.init_sensor_model()

        # dynamics noise for the state (x,y,yaw)
        s.Q = Q

        # we resample particles if the effective number of particles
        # falls below s.resampling_threshold*num_particles
        s.resampling_threshold = resampling_threshold

        # initialize the map
        s.map = map_t(resolution)

    def bresenham(s, p0, p1):
        """
        Ref: Adapted from Peter Corke's implementation of bresenham for his spatial math lib        
        """
        x0, y0 = p0.flatten()
        x1, y1 = p1.flatten()
        dx = x1 - x0
        dy = y1 - y0
        if abs(dx) >= abs(dy):
            # Making line: y = mx + c
            if dx == 0:
                # case p0 == p1
                x = np.array([x0])
                y = np.array([y0])
                
            else:
                c = y0 - (dy / dx) * x0
                if dx > 0:
                    # line to the right
                    x = np.arange(x0, x1 + 1)
                elif dx < 0:
                    # line to the left
                    x = np.arange(x0, x1 - 1, -1)
                y = np.round((dy/dx)*x + c)
        else:
            # Making line x = my + c
            c = x0 - (dx / dy)*y0
            if dy > 0:
                # line to the right
                y = np.arange(y0, y1 + 1)
            elif dy < 0:
                # line to the left
                y = np.arange(y0, y1 - 1, -1)
            x = np.round((dx / dy)*y + c)

        return x.astype(int), y.astype(int)

    def read_data(s, src_dir, idx=0, split='train'):
        """
        src_dir: location of the "data" directory
        """
        logging.info('> Reading data')
        s.idx = idx
        s.lidar = load_lidar_data(os.path.join(src_dir,
                                               'data/%s/%s_lidar%d'%(split,split,idx)))
        s.joint = load_joint_data(os.path.join(src_dir,
                                               'data/%s/%s_joint%d'%(split,split,idx)))

        # finds the closets idx in the joint timestamp array such that the timestamp
        # at that idx is t
        s.find_joint_t_idx_from_lidar = lambda t: np.argmin(np.abs(s.joint['t']-t))

    def init_sensor_model(s):
        # lidar height from the ground in meters
        s.head_height = 0.93 + 0.33
        s.lidar_height = 0.15

        # dmin is the minimum reading of the LiDAR, dmax is the maximum reading
        s.lidar_dmin = 1e-3
        s.lidar_dmax = 30
        s.lidar_angular_resolution = 0.25
        # these are the angles of the rays of the Hokuyo
        s.lidar_angles = np.arange(-135,135+s.lidar_angular_resolution,
                                   s.lidar_angular_resolution)*np.pi/180.0

        # sensor model lidar_log_odds_occ is the value by which we would increase the log_odds
        # for occupied cells. lidar_log_odds_free is the value by which we should decrease the
        # log_odds for free cells (which are all cells that are not occupied)
        s.lidar_log_odds_occ = np.log(9)
        s.lidar_log_odds_free = np.log(1/9.)

    def init_particles(s, n=100, p=None, w=None, t0=0):
        """
        n: number of particles
        p: xy yaw locations of particles (3xn array)
        w: weights (array of length n)
        """
        s.n = n
        s.p = deepcopy(p) if p is not None else np.zeros((3,s.n), dtype=np.float64)
        s.w = deepcopy(w) if w is not None else np.ones(n)/float(s.n)

    @staticmethod
    def stratified_resampling(p, w):
        """
        resampling step of the particle filter, takes p = 3 x n array of
        particles with w = 1 x n array of weights and returns new particle
        locations (number of particles n remains the same) and their weights
        """
        #### TODO: XXXXXXXXXXX
        
        # The resampling does the same thing as stratified... I just made a formula for 
        # frequency and then repeated variables by that frequency to avoid for loop
        # there are edge cases when no. elments are 99 due to int conversion, there I add
        # an extra particle to the one with highest weight

        new_particle_frequency = (np.array(w*(w.shape[0]+2))).astype(int)
        sum_of_freq = new_particle_frequency.sum()
        if sum_of_freq<100:
            new_particle_frequency[np.argmax(new_particle_frequency)] += 100 - sum_of_freq
        newParticles = np.repeat(p,new_particle_frequency,axis=1)[:,:w.shape[0]]
        return newParticles,np.ones_like(w)/w.shape[0]

    @staticmethod
    def log_sum_exp(w):
        return w.max() + np.log(np.exp(w-w.max()).sum())

    def rays2world(s, p, d, head_angle=0, neck_angle=0, angles=None):
        """
        p is the pose of the particle (x,y,yaw)
        angles = angle of each ray in the body frame (this will usually
        be simply s.lidar_angles for the different lidar rays)
        d = is an array that stores the distance of along the ray of the lidar, for each ray (the length of d has to be equal to that of angles, this is s.lidar[t]['scan'])
        Return an array 2 x num_rays which are the (x,y) locations of the end point of each ray
        in world coordinates
        """
        #### TODO: XXXXXXXXXXX not anymore!

        # make sure each distance >= dmin and <= dmax, otherwise something is wrong in reading
        # the data
        # d = np.clip(d, s.lidar_dmin, s.lidar_dmax)  #Should we clip or discard!?
        indices = np.where((d < s.lidar_dmax*0.95) & (d > s.lidar_dmin))
        angles = s.lidar_angles[indices]
        d = d[indices]

        # 1. from lidar distances to points in the LiDAR frame
        #-135 to 135 with centered around zero s.lidar_angular_resolution
        dx = d*np.cos(angles).reshape(1,-1) #-135 to 135
        dy = d*np.sin(angles).reshape(1,-1)
        xy_lidar = np.vstack( (np.vstack((dx,dy)),np.zeros_like(dx)) )

        # 2. from LiDAR frame to the body frame
        H1 = euler_to_se3(0, head_angle, neck_angle, np.array([0,0,s.lidar_height]) )
        # 3. from body frame to world frame
        H2 = euler_to_se3(0,0,p[-1],np.array([p[0],p[1],s.head_height]) )

        xy_lidar = np.vstack( (xy_lidar,np.ones_like(dx)) )        
        xy_world = H2@H1@xy_lidar

        #Avoiding ground detection as obstacles
        xy_world = xy_world[:,xy_world[-1]>=0.05]

        return xy_world[:2,:]

    def get_control(s, t):
        """
        Use the pose at time t and t-1 to calculate what control the robot could have taken
        at time t-1 at state (x,y,th)_{t-1} to come to the current state (x,y,th)_t. We will
        assume that this is the same control that the robot will take in the function dynamics_step
        below at time t, to go to time t-1. need to use the smart_minus_2d function to get the difference of the two poses and we will simply set this to be the control (delta x, delta y, delta theta)
        """

        if t == 0:
            return np.zeros(3)

        pose_t = s.lidar[t]['xyth']
        pose_t_prev = s.lidar[t-1]['xyth']
        return smart_minus_2d(pose_t,pose_t_prev)

    def dynamics_step(s, t):
        """"
        Compute the control using get_control and perform that control on each particle to get the updated locations of the particles in the particle filter, remember to add noise using the smart_plus_2d function to each particle
        """
        u = s.get_control(t)   
        #s.p -> 3xn
        nParticles = s.p.shape[1]
        noise = np.random.multivariate_normal(np.zeros((s.Q.shape[0],)),s.Q,(nParticles))
        s.p = np.array([smart_plus_2d(k,u) for k in s.p.T]).T
        s.p = np.array([smart_plus_2d(s.p[:,i],noise[i,:]) for i in range(nParticles)]).T

    @staticmethod
    def update_weights(w, obs_logp):
        """
        Given the observation log-probability and the weights of particles w, calculate the
        new weights as discussed in the writeup. Make sure that the new weights are normalized
        """
        #### TODO: XXXXXXXXXXX
        wt1 = np.log(w) + obs_logp 
        wt1 -= slam_t.log_sum_exp(wt1)
        return np.exp(wt1)      #we are supposed to use log sum trick

    def observation_step(s, t):
        """
        This function does the following things
            1. updates the particles using the LiDAR observations
            2. updates map.log_odds and map.cells using occupied cells as shown by the LiDAR data

        Some notes about how to implement this.
            1. As mentioned in the writeup, for each particle
                (a) First find the head, neck angle at t (this is the same for every particle)
                (b) Project lidar scan into the world frame (different for different particles)
                (c) Calculate which cells are obstacles according to this particle for this scan,
                calculate the observation log-probability
            2. Update the particle weights using observation log-probability
            3. Find the particle with the largest weight, and use its occupied cells to update the map.log_odds and map.cells.
        You should ensure that map.cells is recalculated at each iteration (it is simply the binarized version of log_odds). map.log_odds is of course maintained across iterations.
        """
        #### TODO: XXXXXXXXXXX
        # raise NotImplementedError It is IMPLEMENTED
        #Get the data at current time stamp
        xyth = s.lidar[t]['xyth']
        d = s.lidar[t]['scan']
        

        tstpm = s.find_joint_t_idx_from_lidar(t)
        neck_angle,head_angle = s.joint['head_angles'][:,tstpm].flatten()

        #First Project Lidar Scan into world Frame
        log_probs = np.zeros(s.p.shape[1])
        for i,p in enumerate(s.p.T):
            occ_cells = s.rays2world(p.T,d,head_angle=head_angle,neck_angle=neck_angle)
            # calculate the observation log-probability
            idx = s.map.grid_cell_from_xy(occ_cells[0,:],occ_cells[1,:])
            log_probs[i] = np.sum(s.map.cells[idx[0],idx[1]])
        #Update the particle weights using observation log-probability
        s.w = s.update_weights(s.w,log_probs)

        #Find the particle with the largest weight 
        idx = int(np.argmax(s.w))
        largest_wgt_pt = s.p[:,idx]
        s.mama_particle = largest_wgt_pt

        #use its occupied cells to update the map.log_odds and map.cells.
        occ_pos = s.rays2world(largest_wgt_pt,d,head_angle=head_angle,neck_angle=neck_angle)
        cells = s.map.grid_cell_from_xy(occ_pos[0,:],occ_pos[1,:]).T
        s.map.log_odds[cells[:,0],cells[:,1]] += s.lidar_log_odds_occ*15
        particle_cell = s.map.grid_cell_from_xy(largest_wgt_pt[0],largest_wgt_pt[1])
        s.map.num_obs_per_cell[cells[:,0],cells[:,1]] = 1

        # How Can we even vectorize this   
        freespace = np.array([0])
        for i in range(cells.shape[0]):
            free_x,free_y = s.bresenham(particle_cell,cells[i,:])
            if freespace.shape[0]==1:
                freespace = np.array([free_x[1:-2],free_y[1:-2]])
            else:
                freespace = np.column_stack((freespace,np.array([free_x[1:-2],free_y[1:-2]])))
        #Unique Pixels to take since we dont want to bias by the log odds
        arr = np.unique(freespace.T,axis=0)
        s.map.log_odds[arr[:,0],arr[:,1]] += s.lidar_log_odds_free*5
        #Clip
        s.map.log_odds = np.clip(s.map.log_odds, -s.map.log_odds_max, s.map.log_odds_max)
        s.map.cells[s.map.log_odds >= s.map.log_odds_thresh] = 1
        s.map.cells[s.map.log_odds < s.map.log_odds_thresh] = 0
        #The sole job of this is to map the brown area and increase computation time
        s.map.num_obs_per_cell[arr[:,0],arr[:,1]] = 1
        s.resample_particles()

    def resample_particles(s):
        """
        Resampling is a (necessary) but problematic step which introduces a lot of variance
        in the particles. We should resample only if the effective number of particles
        falls below a certain threshold (resampling_threshold). A good heuristic to
        calculate the effective particles is 1/(sum_i w_i^2) where w_i are the weights
        of the particles, if this number of close to n, then all particles have about
        equal weights and we do not need to resample
        """
        
        e = 1/np.sum(s.w**2)
        logging.debug('> Effective number of particles: {}'.format(e))
        if e/s.n < s.resampling_threshold:
            s.p, s.w = s.stratified_resampling(s.p, s.w)
            logging.debug('> Resampling')