import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import sys
import gym
import math
import time
import pylab
import cmath
import random
import itertools
import threading
import numpy as np
import pandas as pd
from gym import Env
import scipy.io as sc
import scipy.linalg
from sys import version
from absl import logging
from numpy import ndarray
from scipy import special
from gym.utils import seeding
from scipy.constants import *
from scipy.special import erfinv
from scipy.integrate import quad
from scipy.linalg import toeplitz
from numba import jit, njit, prange
from gym.spaces import Discrete, Box
from collections import deque, Counter
from mpl_toolkits.mplot3d import Axes3D
from gym import Env, error, spaces, utils

import tensorflow as tf
tf.config.run_functions_eagerly(True)  # Enable eager execution

from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Input, Dropout, BatchNormalization

# Set of Times New Roman font
from matplotlib import cm
from matplotlib import colors
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

dtype = np.float32

class M_MIMOEnv(Env):
  def __init__(self, N, M, K):
##########################################################################################################################################################################################################################################
# start : multi-cell massive MIMO envrionment

    self.N = N              # number of cells is equals to number of BSs
    self.M = M              # number of BS transmission antennas
    self.K = K              # number of UEs in a cell
    self.BW = 10e6          # Bandwidth = 10MHz
    self.NF = 7             # Power of noise figure [dBm]
    self.Ns = 10            # Number of sample
    self.min_p = -20        # Minimum transmission power [dBm]
    self.max_p = 23         # Maximum transmission power [dBm]
    self.num_p = 10         # Number of possible actions

    # get multi-cell massive MIMO channel matrix (slow generating w/ shadowing, fading)
    self.H, self.H_gain = self.get_channel_local_scatter(no_realization = self.Ns)

    # get ZF precoding vector without antenna selection
    self.precoding = self.antenna_selection(self.H, ant_sel = False)

    # get random seed
    self.seed()

    # initialize step counter
    self.count = 0

    # initialize state
    self.state = None
    self.downlink_rate = None
    self.rate_list = []

    self.action_values = self.get_power_set()
    self.num_actions = len(self.action_values)
    self.action_space = spaces.Discrete(self.num_actions)
    self.action_length = self.num_actions

    # observation consists of the transmission power, SINR, and average sum-rate
    lower_bound = np.array([self.min_p, -(np.finfo(np.float32).max), -(np.finfo(np.float32).max)], dtype=np.float32)
    higher_bound = np.array([self.max_p, np.finfo(np.float32).max, np.finfo(np.float32).max], dtype=np.float32)

    self.observation_space = spaces.Box(lower_bound, higher_bound, dtype = np.float32)



##########################################################################################################################################################################################################################################
# start : multi-cell massive MIMO downlink scenario

  # generate dBm
  def get_power_set(self):
    # Generate power levels from min_p to max_p in dBm
    power_set = np.linspace(self.min_p, self.max_p, self.num_p)
    # Convert dBm to watts
    power_set = 1e-3 * np.power(10., power_set/10.)
    return power_set

  # generate random seed
  def randn2(self, *args, **kargs):
    args_r = tuple(reversed(args))
    uniform = np.random.rand(*args_r)
    untiform = uniform.transpose()

    return np.sqrt(2) * erfinv(2 * uniform - 1)

  # generate hermitian matrix
  def hermitian(self, X):
    return X.conj().swapaxes(-1, -2)
  # Divide function by using hermitian.
  def mldivide(self, A: ndarray, B: ndarray, A_is_hermitian=False):
    if A_is_hermitian:
        return self.hermitian(np.linalg.solve(A, self.hermitian(B)))
    else:
        return self.hermitian(np.linalg.solve(self.hermitian(A), self.hermitian(B)))


  # correlation in term of real. signal parts.
  @njit
  def correlation_real(self, x, antenna_spacing, col):
    return np.cos(2 * np.pi * antenna_spacing * col * np.sin(x))


  # correlation in term of imag. signal parts.
  @njit
  def correlation_imag(self, x, antenna_spacing, col):
    return np.sin(2 * np.pi * antenna_spacing * col * np.sin(x))


  # probability density function (PDF) : gussian
  @njit
  def gaussian_pdf(self, x, mean, dev):
    return np.exp(-(x-mean) ** 2 / (2 * dev ** 2)) / (np.sqrt(2 * np.pi) * dev)


  # correlation function
  @njit
  def corr(self, x, theta, asd, antenna_spacing, dist, col, real_imag):
    if real_imag == 0:
        res = np.cos(2 * np.pi * antenna_spacing * col * np.sin(x))
    else:
        res = np.sin(2 * np.pi * antenna_spacing * col * np.sin(x))
    if dist =='gaussian':
        res *= self.gaussian_pdf(x, theta, asd)
    return res


  # Convert dBm to watts
  def dBm2Watts(self, P):
    # 1e-3* pow(10., self.n_power/10.)
    # P[w] = 10 ^ ((P[dBm] - 30) / 10);
    # P_watts = 1e-3 * pow(10., P/10.)
    P_watts = (10 ** ((P - 30) / 10))
    return P_watts


  # local scattering channel model. where the local is a single cell in total cell
  def R_local_scattering(self, M, theta, asd_deg, antenna_spacing=0.5, dist='Gaussian', accuracy=1, dtype=np.complex128):

    # In radians
    asd = asd_deg * np.pi / 180

    # correlation matrix is Toeplitz structure, so only need first row
    first_row = np.zeros([M,], dtype=dtype)

    if accuracy == 1:
        lb = None
        ub = None

        dist = dist.lower()
        if dist == 'gaussian':
            lb = theta - 20 * asd
            ub = theta + 20 * asd

        else:
            raise NotImplementedError

        for col in range(0, M):
            # distance from the first antenna
            c_real:float = quad(func=self.corr, a=lb, b=ub, args=(theta, asd, antenna_spacing, dist, col, 0))[0]
            c_imag:float = quad( func=self.corr, a=lb, b=ub, args=(theta, asd, antenna_spacing, dist, col, 1))[0]

            first_row[col] = complex(c_real, c_imag)
    elif accuracy == 2:
        # Gaussian distribution
        distance = np.arange(M)
        x1 = np.exp(1j * 2 * np.pi * antenna_spacing * np.sin(theta) * distance)
        x2 = np.exp(-asd ** 2 / 2 * (2 * np.pi * antenna_spacing * np.cos(theta) * distance) ** 2)
        first_row = x1 * x2

    return toeplitz(c=first_row.conjugate())


  # Channel statistics between UE's at random locations and the BS.
  def channel_stat_setup(self, N, K, M, asd_degs, no_BS_per_dim=None, accuracy=2,):
    # square side, in meters
    side_length = 500

    # pathloss exp
    alpha = 3.76

    # avg. channel gain in dB at the ref. distance 1 meter. At exponent set to 3.76, at 1km it's -148.1 dB
    constant_term = -35.3

     # standard deviation of shadow fading
    sigma_sf = 10

    # minimum distance between BS and UEs
    min_UE_BS_dist = 25

    # maximum distance between BS and UEs
    max_UE_BS_dist = 300

    # antenna spacing # of wavelengths
    antenna_spacing = 0.5
    if no_BS_per_dim is None:
        no_BS_per_dim = np.array([np.sqrt(N), np.sqrt(N)])
    inter_bs_distance = side_length / no_BS_per_dim

    # scatter the BSs
    BS_positions = np.stack(np.meshgrid(np.arange(inter_bs_distance[0]/2, side_length, inter_bs_distance[0]),np.arange(inter_bs_distance[1]/2, side_length, inter_bs_distance[1]),indexing='ij'),axis=2).reshape([-1,2])

    # now all the other nine alternatives of the BS locations
    wrap_locations = np.stack(np.meshgrid(np.array([-side_length, 0, side_length]), np.array([-side_length, 0, side_length]), indexing='ij' ), axis=2).reshape([-1,2])

    # for each BS locations, there are 9 possible alternative locations including the original one. Here uses broadcasting to add (9,2) to a (num_BS, 1, 2) to get a (num_BS, 9, 2)
    BS_positions_wrapped = np.expand_dims(BS_positions, axis=1) + wrap_locations

    UEpositions = np.zeros([K, N, 2])
    perBS = np.zeros([N,], dtype=np.int32)

    # normalized spatial correlation matrices
    R = np.zeros([M, M, K, N, N, len(asd_degs)], dtype=np.complex128)

    self.channel_gain = np.zeros([K, N, N])

    for i in range(N):
        # put K UEs in the cell, uniformly. UE's not satisfying the minimum distance are replaced
        res = []
        while perBS[i] < K:
            UEremaining = K - perBS[i]
            pos = np.random.uniform(-inter_bs_distance/2, inter_bs_distance/2,size=(UEremaining, 2))
            cond = np.linalg.norm(pos, ord=2, axis=1) >= min_UE_BS_dist

            # satisfying minimum distance with respect to BS shape
            pos = pos[cond, :]
            for x in pos:
                res.append(x + BS_positions[i])
            perBS[i] += pos.shape[0]
        UEpositions[:, i, :] = np.array(res)

        # loop through all BS for cross-channels
        for j in range(N):
            # distance between all UEs in cell i to BS j, considering wrap-around.
            dist_ue_i_j = np.linalg.norm(np.expand_dims(UEpositions[:, i], axis=1) - BS_positions_wrapped[j, :, :], axis=2)
            dist_bs_j = np.min(dist_ue_i_j, axis=1)
            which_pos = np.argmin(dist_ue_i_j, axis=1)

            # average channel gain with large-scale fading mdoel in (2.3), neglecting shadow fading
            self.channel_gain[:, i, j] = constant_term - alpha * 10 * np.log10(dist_bs_j)

            # generate spatial correlation matrices for channels with local scattering model
            for k in range(K):
                vec_ue_bs = UEpositions[k, i] - BS_positions_wrapped[j, which_pos[k]]
                angle_BS_j = np.arctan2(vec_ue_bs[1], vec_ue_bs[0])
                for spr, asd_deg in enumerate(asd_degs):
                    R[:, :, k, i, j, spr] = self.R_local_scattering( M, angle_BS_j, asd_deg, antenna_spacing, accuracy=accuracy)

        # all UEs in cell i to generate shadow fading realizations
        for k in range(K):
            # see if another BS has a larger avg. channel gain to the UE than BS i
            while True:
                # generate new shadow fading realizations until all UE's in cell i has its largest avg. channel gain from BS i
                shadowing = sigma_sf * self.randn2(N)
                channel_gain_shadowing = self.channel_gain[k, i] + shadowing
                if channel_gain_shadowing[i] >= np.max(channel_gain_shadowing):
                    break
            self.channel_gain[k,i,:] = channel_gain_shadowing

    return R, self.channel_gain


  # Normalized the complex values.
  def complex_normalize(self, X, axis=-1):
    mags = np.linalg.norm(np.abs(X), axis=axis, keepdims=True)
    return X / mags


  # Generate noise figure with bandwidth (20MHz)
  def noise_dbm(self,):
    return -174 + 10 * np.log10(self.BW) + self.NF


  # generate the ZF precoded combination
  def zf_combining(self,H):
    H1 = H
    A = self.hermitian(H1) @ H1 + 1e-12 * np.eye(H1.shape[-1])
    B = H1
    res = self.mldivide(A, B, A_is_hermitian=True)
    return res


  # Generate uncorrelated Rayleigh fading channel realizations with unit variance
  def get_H_rayleigh_unit(self, M, K, N, Ns):
    randn2 = np.random.randn
    H = randn2(N, N, K, M, Ns) + 1j * randn2(N, N, K, M, Ns)
    return np.sqrt(0.5) * H


  # Get local scattering channel matrix
  def get_channel_local_scatter(self, no_realization):
    # return shape:  Ns x N x N x K x M
    if self.N > 1 and self.N < 4:
        no_BS_per_dim = np.array([1, self.N])
    else:
        no_BS_per_dim = None
    R, gain_db = self.channel_stat_setup(self.N, self.K, self.M,no_BS_per_dim=no_BS_per_dim,asd_degs=[30,], accuracy=2)
    gain_db -= self.noise_dbm()

    # shape is M x M x K x N x N x no_asd_degs
    R_gain = R[:, :, :, :, :, 0] * np.power(10, gain_db / 10.0)
    R_gain = np.ascontiguousarray(np.transpose(R_gain[:, :, :, :, :], (4,3,2,1,0)))

    # now the shape is N x N x K x M x M
    # for each user, the channel between some BS to it, what is the spatial correlation.
    # Therefore in total there are so many numbers: K * N * N * M * M
    H = self.get_H_rayleigh_unit(self.M, self.K, self.N, self.Ns)
    H_gain = np.zeros_like(H)
    for _idx in np.ndindex(*H.shape[0:3]):
        H_gain[_idx] = scipy.linalg.sqrtm(R_gain[_idx]) @ H[_idx]
        # TODO: use Cholesky factorization to replace the slow matrix square root operation.
        # However, it requires the matrix to be positive semidefinite, which should be the case but due to the numerical error is not always the case.
    res = np.ascontiguousarray(np.transpose(H_gain, (4, 0, 1, 2, 3)))
    return res, H_gain


  # setting of the ZF precoding method
  def get_precoding(self, H, method="ZF", local_cell_info=True):
    res = []
    if method == "ZF":
        algo = self.zf_combining
    if local_cell_info:
        no_cells = H.shape[1]
        for j in range(no_cells):
            res.append(algo(H[:,j,j]))
    return np.stack(res, axis=1)


  # Antenna selection and power allocation by using ZF precoding
  def antenna_selection(self, H, ant_sel=True):
    Ns, N, N, K, M = H.shape
    if ant_sel:

        antenna_sel = np.zeros((Ns, N, M), dtype=np.bool_)

        # strongest K_0 antennas
        K_0 = int(M * 0.8)
        for r in range(Ns):
            for n in range(N):
                channel_power_ant = (np.abs(H[r, n, n]) ** 2).sum(axis=-2)
                top_k = np.argsort(channel_power_ant)[0:K_0]
                antenna_sel[r, n][top_k] = True
        # or randomly
        H_n = (H.transpose(2, 3, 0, 1, 4) * antenna_sel).transpose(2, 3, 0, 1, 4)

    else:
        antenna_sel = np.ones((Ns, N, M), dtype=np.bool_)
        H_n = H
    W = self.get_precoding(H_n, method="ZF", local_cell_info=True)

    # power allocation
    # W = np.sqrt(self.config["P_DL"]) * self.complex_normalize(W)
    # return W, antenna_sel
    return W


  # Calculate the downlink user-rate [bits/s/Hz]
  def DL_rate(self, channel, precoding, power):
    H, V = channel, precoding
    W = self.complex_normalize(V, -1)

    Interval, N, K, M = H.shape[0], H.shape[1], H.shape[3], H.shape[4]
    intercell_intf = np.zeros((N, K))
    intracell_intf = np.zeros((Interval, N, K))
    sig = np.zeros((Interval, N, K))

    for n in range(Interval):
        for l in range(N):
            H_l = H[n, l] # (N, K, M)
            for k in range(K):
                w_l = W[n, l] # (K, M)
                H_llk = H_l[l, k] # (M, ) the channel between l-th BS to user k
                p_l = np.abs(np.dot(w_l.conj(), H_llk)) ** 2
                sig[n, l, k] = p_l[k]
                intracell_intf[n, l, k] = p_l.sum() - p_l[k]
                if N > 1:
                    idx_othercell = list(range(N))
                    idx_othercell.remove(l)
                    H_intercell = H[n, idx_othercell, l:l+1, k] # (L-1, 1, M) CSI, other cells to this user k
                    w_intercell = W[n, idx_othercell] #(L-1, K, M) other cell's precoding vector
                    p_inter = np.abs(w_intercell @ (H_intercell.swapaxes(-1, -2))) ** 2
                    intercell_intf[l,k] += p_inter.sum() / Interval

    int_noise = power * intercell_intf + power * intracell_intf + 1
    self.sinr = (power * sig / int_noise)


    downlink_rate = np.log2(1+self.sinr).mean(axis=0)
    # self.downlink_rate = self.BW * np.log2(1 + self.sinr).mean(axis = 0)
    return downlink_rate

##########################################################################################################################################################################################################################################
# start : reinforcement function
  def seed(self, seed = None):
    self.np_random = seed = seeding.np_random(seed)
    return [seed]

  def step(self, action):
      # Select the power corresponding to the action
      power = self.action_values[action]

      # Calculate downlink rates and SINR for all users
      downlink_rate = self.DL_rate(self.H, self.precoding, power)

      # Compute SINR for all users
      sinr = self.sinr

      # Calculate the total sum-rate as the reward
      reward = self.BW * np.sum(np.log2(1 + sinr))

      # Update the state: power, mean SINR, and total sum-rate
      s_power = power
      s_sinr = np.mean(sinr)
      s_sumrate = np.sum(downlink_rate)

      self.state = (s_power, s_sinr, s_sumrate)

      # Decrement the number of actions remaining
      self.action_length -= 1

      # Check if the episode is done
      done = self.action_length <= 0

      # Increment step count
      self.count += 1

      # Info dictionary for debugging
      info = {}

      return np.array(self.state), reward, done, info, downlink_rate



  def reset(self):
    # mixing the random seed and select the random transmit power
    self.seed()

    # we consider the dynamic changes the UEs location, and shadowing, fading... etc.
    # get multi-cell massive MIMO channel matrix (slow generating w/ shadowing, fading)
    self.H, self.H_gain = self.get_channel_local_scatter(no_realization = self.Ns)

    # get ZF precoding vector without antenna selection
    self.precoding = self.antenna_selection(self.H, ant_sel = False)

    # reset the step counter
    self.count = 0

    # allocate random value each s_power, s_sinr, s_sumrate
    self.state = np.random.uniform(low = -0.05, high = 0.05, size = (3,))

    downlink_rate = None

    self.rate_list = []

    self.action_length = self.num_actions

    # return self.state
    # return np.array(self.state, dtype = np.float32)
    return np.array(self.state)

  # Equal Power Allocation
  def equal_PA(self):
      # Maximum downlink transmit power per BS (mW)
      pmax = self.dBm2Watts(self.max_p)
      rhoEqual = (pmax / self.K) * np.ones((1, self.K))

      # Compute downlink power per UE in case of equal power allocation
      equal_power = np.sum(np.sqrt(rhoEqual))
      downlink_rate = self.DL_rate(self.H, self.precoding, equal_power)

      # Compute the total sum-rate (similar to reward in DQN)
      total_sum_rate = self.BW * np.sum(np.log2(1 + self.sinr))

      return total_sum_rate

  # Randomly Power Allocation
  def random_PA(self):
      random_power = np.sum(
          np.random.uniform(self.dBm2Watts(self.min_p), self.dBm2Watts(self.max_p), size=self.K)
      )
      downlink_rate = self.DL_rate(self.H, self.precoding, random_power)

      # Compute the total sum-rate
      total_sum_rate = self.BW * np.sum(np.log2(1 + self.sinr))

      return total_sum_rate

  # Maximum Power Allocation
  def maximum_PA(self):
      max_power = np.sum(self.dBm2Watts(self.max_p) * np.ones((self.K)))
      downlink_rate = self.DL_rate(self.H, self.precoding, max_power)

      # Compute the total sum-rate
      total_sum_rate = self.BW * np.sum(np.log2(1 + self.sinr))

      return total_sum_rate

  # Simplified Maximum Product Power Allocation
  def maxprob_PA(self):
      R_list = []
      P_list = []
      SP_list = []

      for j in range(self.N):
          for i in range(self.K):
              i_maxprob_power = np.sum(self.dBm2Watts(self.max_p) * np.ones((self.M)))
              i_maxprob_rate = self.DL_rate(self.H, self.precoding, i_maxprob_power).sum() / self.N

              P_list.append(i_maxprob_power)
              R_list.append(i_maxprob_rate)

              i_rate = P_list
              i_idx = list(np.arange(len(i_rate)))
              maxprob_dict = dict(zip(i_idx, i_rate))

          # Extract optimal transmission power
          op_power = max(maxprob_dict, key=maxprob_dict.get)

          SP_list.append(P_list[op_power])

      est_power = np.sum(SP_list)
      downlink_rate = self.DL_rate(self.H, self.precoding, est_power)

      # Compute the total sum-rate
      total_sum_rate = self.BW * np.sum(np.log2(1 + self.sinr))

      return total_sum_rate

if __name__ == "__main__":
    # Initialize environment with parameters
    N = 7  # number of cells
    M = 100  # number of antennas
    K = 10  # number of users per cell
    
    # Create environment
    env = M_MIMOEnv(N, M, K)
    
    # Test reset
    print("\nTesting environment reset...")
    initial_state = env.reset()
    print(f"Initial state: {initial_state}")
    
    # Test different power allocation methods
    print("\nTesting power allocation methods...")
    print(f"Equal Power Allocation Sum-Rate: {env.equal_PA():.2f} bits/s")
    print(f"Random Power Allocation Sum-Rate: {env.random_PA():.2f} bits/s")
    print(f"Maximum Power Allocation Sum-Rate: {env.maximum_PA():.2f} bits/s")
    print(f"MaxProb Power Allocation Sum-Rate: {env.maxprob_PA():.2f} bits/s")
    
    # Test step function
    print("\nTesting step function...")
    action = 0  # Test with first action
    next_state, reward, done, info, dl_rate = env.step(action)
    print(f"Next state: {next_state}")
    print(f"Reward: {reward:.2f}")
    print(f"Done: {done}")
    print(f"Downlink Rate: {dl_rate}")