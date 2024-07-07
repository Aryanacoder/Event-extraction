#! /usr/bin/env python

"""
Apply POD to CUBE flow field data

Ref: Kunihiko TAIRA, Proper Orthogonal Decomposition in Fluid Flow Analysis: 1. Introduction, Nagare, 115-123, 30 (2011)
"""

import os, sys; sys.path.append('../src/common')
import logging
import numpy as np
import matplotlib
import matplotlib.cm as cm
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mpl_toolkits.axes_grid1
import toml

from util import read_input
from parameter import Parameter
from preprocess import preprocess

from log import Logger

import cProfile
import pstats

from memory_profiler import profile

@profile
def main():
    args = sys.argv
    if len(args) == 1:
        raise Exception("Specify configuration file!")
    elif len(args) >= 3:
        raise Exception("Too many configuration files is specified!")
    fpath_param = sys.argv[1]
    
    nproc = 1
    rank = 0
    
    # Make required directory
    dirpath_input = '../results/xmean'
    dirpath_output = '../results/pod_solve'
    dirpath_img = '../img/pod_solve'
    dirpath_log = '../log/pod_solve'
    os.makedirs(dirpath_output, exist_ok=True)
    os.makedirs(dirpath_img, exist_ok=True)
    os.makedirs(dirpath_log, exist_ok=True)
    
    param = Parameter(fpath_param)
    preprocess(param)
    
    logger = Logger(dirpath_log, param, rank).logger
    
    width = param.xend-param.xsta
    height = param.yend-param.ysta
    depth = param.zend-param.zsta
    logger.debug('width = {0}, height = {1}, depth = {2}'.format(width, height, depth))
    
    # Load flow field
    # X: Original flow field
    X = read_input(param.step_sta, param.step_end, param.step_spc, width, height, depth, param.ndim, dirpath=param.dirpath_flowfield, na_header=param.na_header)
    
    if (X.shape[-1] == param.ndim + 1) and (param.use_pressure == False):
        X = X[...,1:]
    logger.debug(f'Shape of X = {X.shape}')
    
    # Xmean: Time mean value of original flow field
    #Xmean = X.mean(axis=0)
    logger.info('Load Xmean ...')
    with open(os.path.join(dirpath_input, 'xmean.ary'), 'rb') as f:
        if param.ndim == 2:
            Xmean = np.fromfile(f, np.float32).reshape(width, height, -1)
        elif param.ndim == 3:
            Xmean = np.fromfile(f, np.float32).reshape(width, height, depth, -1)
    
    if (Xmean.shape[-1] == param.ndim + 1) and (param.use_pressure == False):
        Xmean = Xmean[...,1:]
    logger.debug(f'Shape of Xmean = {Xmean.shape}')
    
    # X: fluctuation component of flow field (Eq.10)
    X = (X - Xmean).reshape(X.shape[0],-1).transpose(1,0)
    
    # R: Covariance matrix for Snapshot POD (Transposition of Eq.11)
    logger.info('Make covariance matrix')
    R = (X.T).dot(X)
    
    # Profiling (Start)
    pr = cProfile.Profile()
    pr.enable()
    
    # Solve eigenvalue problem of Snapshot POD (Eq.13)
    # lambda_: Eigenvalue
    # u: Eigenvector
    logger.info('Solve eigenvalue problem: Martix shape = {0}'.format(R.shape))
    lambda_, u = np.linalg.eigh(R)
    
    # Profiling (End)
    pr.disable()
    stats = pstats.Stats(pr)
    print("sort by cumtime")
    stats.sort_stats('cumtime')
    stats.print_stats()
    
    # Sort in descending order
    lambda_ = lambda_[::-1]
    u = u[:, ::-1]
    
    # Make graph of eigenvalue of each mode
    fig = plt.figure()
    plt.plot(lambda_[:10])
    plt.savefig(os.path.join(dirpath_img, 'eigenvalues.png'))
    plt.clf()
    
    # Make graph of contribution of each mode
    plt.plot([lambda_[:i].sum() / lambda_.sum() for i in range(len(lambda_[:10]))])
    plt.savefig(os.path.join(dirpath_img, 'contribution.png'))
    
    # Get index in which eigenvalue is positive
    r = len(lambda_[lambda_>0])
    
    # phi: POD basis (Eq.14)
    logger.info('Calculate POD basis')
    X = X.dot(u[:,:r]/np.sqrt(lambda_[:r]))
    
    # Output eigenvalue
    logger.info('Outputting eigenvalue ...')
    lambda_[:r].tofile(os.path.join(dirpath_output, 'lambda.ary'))
    
    # Output POD basis
    logger.info('Outputting POD basis ...')
    X.tofile(os.path.join(dirpath_output, 'phi.ary'))
    
    # Make figure of time mean flow field
    #plt.clf()
    #if param.ndim == 2:
    #    Uf0 = Xmean[:,:,0]
    #elif param.ndim == 3:
    #    Uf0 = Xmean[:,:,12,0]
    #mappable = plt.imshow(Uf0.T, vmin=-0.2, vmax=1.4 ,cmap=cm.jet, interpolation='bicubic')
    #fig.colorbar(mappable)
    #plt.savefig(os.path.join(dirpath_img, 'mode0.png'))
    
    # Visualize each POD basis
    #if param.ndim == 2:
    #    mode = phi.reshape(384,192,2,phi.shape[1])
    #elif param.ndim == 3:
    #    mode = phi.reshape(384,192,24,3,phi.shape[1])
    #for i in range(phi.shape[1]):
    #    fig, ax = plt.subplots(1, 1)
    #    if param.ndim == 2:
    #        u_mode = mode[:,:,0,i]
    #    elif param.ndim == 3:
    #        u_mode = mode[:,:,12,0,i]
    #    # mappable = plt.imshow(u_mode.T, vmin=param.list_cb_lower[0], vmax=param.list_cb_upper[0] ,cmap=cm.jet, interpolation='bicubic')
    #    #mappable = ax.imshow(u_mode.T, vmin=param.list_cb_lower[0], vmax=param.list_cb_upper[0] ,cmap=cm.jet, interpolation='bicubic')
    #    mappable = ax.imshow(u_mode.T, vmin=-0.001, vmax=0.001 ,cmap=cm.jet, interpolation='bicubic')
    #    divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax)
    #    cax = divider.append_axes('right', '5%', pad='3%')
    #    fig.colorbar(mappable, ax=ax, cax=cax)
    #    plt.savefig(os.path.join(dirpath_img, 'mode{0:04d}.png'.format(i+1)))
    #    plt.close()
    #    plt.clf()
    #    plt.cla()
    #

main()
