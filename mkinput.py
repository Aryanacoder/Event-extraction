#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright Â© 2019 Kazuto Ando <kazuto.ando@riken.jp>
#
# Distributed under terms of the MIT license.

"""

"""

import logging
import os, sys; sys.path.append('../src/common')

import glob
import numpy as np
import toml

from util import *
from parameter import Parameter
from preprocess import preprocess

from log import Logger

from communicate import *

import mpi4py
from mpi4py import MPI

args = sys.argv
if len(args) == 1:
    raise Exception("Specify configuration file!")
elif len(args) >= 3:
    raise Exception("Too many configuration files is specified!")
fpath_param = sys.argv[1]

comm_world = MPI.COMM_WORLD
nproc_world = comm_world.Get_size()
rank_world = comm_world.Get_rank()

# Make required directory
dirpath_log = '../log/mkinput'
os.makedirs(dirpath_log, exist_ok=True)

param = Parameter(fpath_param)
preprocess(param)

logger = Logger(dirpath_log, param, rank_world).logger

if param.use_pressure == True:
    nq = param.ndim + 1
else:
    nq = param.ndim

color = rank_world // param.cp
key = rank_world
comm_cp = comm_world.Split(color, key)       # Communicator for model parallel
nproc_cp = comm_cp.Get_size()                # Number of processes for model parallel
rank_cp = comm_cp.Get_rank()                 # Rank number for model paralell
logger.debug("nproc_cp = {0}, rank_cp = {1}".format(nproc_cp, rank_cp))

color_dp = rank_world % param.cp
key_dp = rank_world
comm_dp = comm_world.Split(color_dp, key_dp) # Communicator for data parallel
nproc_dp = comm_dp.Get_size()                # Number of processes for data parallel
rank_dp = comm_dp.Get_rank()                 # Rank number for data paralell
logger.debug("nproc_dp = {0}, rank_dp = {1}".format(nproc_dp, rank_dp))

ncellsx = param.xend - param.xsta
ncellsy = param.yend - param.ysta
ncellsz = param.zend - param.zsta

if param.use_pressure == True:
    qsta = 0
else:
    qsta = 1
qend = param.ndim + 1

if len(param.steps) == 0:
    lst_steps = [x for x in range(param.step_sta, param.step_end, param.step_spc)]
else:
    lst_steps = param.steps
lst_steps_decomp = get_decomposed_list_cyclic(lst_steps, nproc_dp, rank_dp)
logger.debug("lst_steps_decomp = {0}".format(lst_steps_decomp))

lst_cube_ids = [x for x in range(param.ncubes)]
lst_cube_ids_decomp = get_decomposed_list_block(lst_cube_ids, nproc_cp, rank_cp)
logger.debug("lst_cube_ids_decomp = {0}".format(lst_cube_ids_decomp))

for istep in lst_steps_decomp: # Flow field snapshots
    fpath = '{0}/field_{1:010d}.bin'.format(param.dirpath_cube_output, istep)
    logger.debug('Input: {0}'.format(fpath))
    with open(fpath) as f:
        f.seek(12)
        # Hierarchical cells used in calculation by CUBE
        na_cells_hi = np.fromfile(f, np.float64, qend*param.ncubes*param.xsize_wh*param.ysize_wh*param.zsize_wh).reshape(qend, param.ncubes, param.zsize_wh, param.ysize_wh, param.xsize_wh)

    # Create equidistant cells
    na_cells_eq = np.zeros(ncellsx*ncellsy*ncellsz*nq).reshape(nq, ncellsz, ncellsy, ncellsx)

    for q in range(qsta, qend): # For 0: p, 1: u, 2: v, 3: w
        #for i in range(na_cells_hi.shape[1]): # ncubes
        for i in lst_cube_ids_decomp: # ncubes
            if i % 1000 == 0:
                logger.info('Timestep, qtype, cube id = {0}, {1}, {2}'.format(istep, q, i))
            na_gridpoint = param.na_gridfile[i]
            if param.na_gridfile.shape[1] == 4: # Single Run-Length
                # Size of this cube
                cubesize_x = na_gridpoint[0]
                cubesize_y = na_gridpoint[0]
                cubesize_z = na_gridpoint[0]
                # Start position of this cube
                xsta = na_gridpoint[1]
                ysta = na_gridpoint[2]
                zsta = na_gridpoint[3]
            elif param.na_gridfile.shape[1] == 6: # Multi Run-Length
                # Size of this cube
                cubesize_x = na_gridpoint[0]
                cubesize_y = na_gridpoint[1]
                cubesize_z = na_gridpoint[2]
                # Start position of this cube
                xsta = na_gridpoint[3]
                ysta = na_gridpoint[4]
                zsta = na_gridpoint[5]
            # Grid index of start point of this cube
            jsta = int((xsta - param.xmin) / param.dx_min)
            ksta = int((ysta - param.ymin) / param.dy_min)
            lsta = int((zsta - param.zmin) / param.dz_min)
            # Ratio of size of this cube to minimum (Enlargement rate)
            ratio_enlarge_x = int(cubesize_x / param.cube_size_min_x)
            ratio_enlarge_y = int(cubesize_y / param.cube_size_min_y)
            ratio_enlarge_z = int(cubesize_z / param.cube_size_min_z)
            # Loop for hierarchical cells (z-direction)
            for il in range(param.halosize, na_cells_hi.shape[2] - param.halosize):
                # Start index of this hierarchical cell (z-direction)
                l1 = lsta + (il - param.halosize) * ratio_enlarge_z
                # End index of this hierarchical cell (z-direction)
                l2 = l1 + ratio_enlarge_z
                # Range inside/outside judgment (z-direction)
                if l1 < param.zend and param.zsta < l2: # Overlap
                    if l1 < param.zsta: # Sticking out to left
                        l1 = param.zsta
                    if param.zend < l2: # Sticking out to right
                        l2 = param.zend
                else: # Not overlap
                    continue
                # Loop for hierarchical cells (y-direction)
                for ik in range(param.halosize, na_cells_hi.shape[3] - param.halosize):
                    # Start index of this hierarchical cell (y-direction)
                    k1 = ksta + (ik - param.halosize) * ratio_enlarge_y
                    # End index of this hierarchical cell (y-direction)
                    k2 = k1 + ratio_enlarge_y
                    # Range inside/outside judgment (y-direction)
                    if k1 < param.yend and param.ysta < k2: # Overlap
                        if k1 < param.ysta: # Sticking out to left
                            k1 = param.ysta
                        if param.yend < k2: # Sticking out to right
                            k2 = param.yend
                    else: # Not overlap
                        continue
                    # Loop for hierarchical cells (x-direction)
                    for ij in range(param.halosize, na_cells_hi.shape[4] - param.halosize):
                        # Start index of this hierarchical cell (x-direction)
                        j1 = jsta + (ij - param.halosize) * ratio_enlarge_x
                        # End index of this hierarchical cell (x-direction)
                        j2 = j1 + ratio_enlarge_x
                        # Range inside/outside judgment (x-direction)
                        if j1 < param.xend and param.xsta < j2: # Overlap
                            if j1 < param.xsta: # Sticking out to left
                                j1 = param.xsta
                            if param.xend < j2: # Sticking out to right
                                j2 = param.xend
                        else: # Not overlap
                            continue
                        # Copy value of this hierarchical cell to equidistant cell
                        na_cells_eq[q - qsta, l1-param.zsta:l2-param.zsta, k1-param.ysta:k2-param.ysta, j1-param.xsta:j2-param.xsta] = na_cells_hi[q, i, il, ik, ij]

    recvbuf = np.empty_like(na_cells_eq)
    comm_cp.Reduce(na_cells_eq, recvbuf)
    na_cells_eq = recvbuf

    if rank_cp == 0:
        # Output equidistant cells in specified range
        fname = '{0}/na_nn_{1:010d}.ary'.format(param.dirpath_flowfield, istep)
        logger.info('Outputting {0} ...'.format(fname))
        if param.ndim == 2:
            #na_cells_eq[param.xsta:param.xend, param.ysta:param.yend, int(param.ncellsz/2), :].astype(np.float32).tofile(fname)
            na_cells_eq[:, int(param.ncellsz/2), :, :].transpose([3, 2, 1, 0]).astype(np.float32).tofile(fname)
        elif param.ndim == 3:
            #na_cells_eq[param.xsta:param.xend, param.ysta:param.yend, param.zsta:param.zend, :].astype(np.float32).tofile(fname)
            na_cells_eq.astype(np.float32).transpose([3, 2, 1, 0]).tofile(fname)
