# -*- coding: utf-8 -*-

from numpy import float32 as npfloat
from numpy import int32 as npint
from numpy import bool as npbool
from numpy import zeros
from numpy import logical_and
from numpy.random import random

# from numpy import pi
# TWOPI = pi*2
# PI = pi
# HPI = pi*0.5


class Automata(object):
  def __init__(
      self,
      grid_size,
      initial,
      influence_rad,
      threads = 256
      ):

    self.itt = 0

    self.threads = threads

    self.grid_size = grid_size # power of two
    # self.one = 1.0/size
    self.influence_rad = influence_rad

    self.__init(initial)
    self.__cuda_init()

  def __init(self, initial):
    grid_size = self.grid_size
    self.total_grid_size = grid_size*grid_size
    self.grid = zeros((grid_size, grid_size), npbool)
    self.massx = zeros((grid_size, grid_size), npfloat)
    self.massy = zeros((grid_size, grid_size), npfloat)
    self.neigh = zeros((grid_size, grid_size), npint)
    self.hits = zeros((grid_size, grid_size), npint)
    self.grid[:,:] = initial

  def _diminish(self, prob):
    ii,jj = logical_and(self.hits<2, self.grid).nonzero()
    diminish_mask = random(size=len(ii))<prob
    self.grid[ii[diminish_mask], jj[diminish_mask]] = False

  def __cuda_init(self):
    import pycuda.autoinit
    from .helpers import load_kernel

    self.cuda_mass = load_kernel(
        'modules/cuda/mass.cu',
        'mass',
        subs={'_THREADS_': self.threads}
        )
    self.cuda_agg = load_kernel(
        'modules/cuda/agg.cu',
        'agg',
        subs={'_THREADS_': self.threads}
        )

  def step(self):
    import pycuda.driver as drv
    self.itt += 1

    grid = self.grid
    blocks = self.total_grid_size//self.threads + 1

    self.cuda_mass(
        npint(self.total_grid_size),
        npint(self.grid_size),
        drv.In(grid),
        npint(self.influence_rad),
        drv.InOut(self.massx[:,:]),
        drv.InOut(self.massy[:,:]),
        drv.Out(self.neigh[:,:]),
        block=(self.threads,1,1),
        grid=(blocks,1)
        )

    self.hits[:,:] = 0

    self.cuda_agg(
        npint(self.total_grid_size),
        npint(self.grid_size),
        drv.In(grid),
        drv.In(self.massx[:,:]),
        drv.In(self.massy[:,:]),
        drv.In(self.neigh[:,:]),
        drv.InOut(self.hits[:,:]),
        block=(self.threads,1,1),
        grid=(blocks,1)
        )

    # self._diminish(0.1)

    hi, hj = self.hits.nonzero()
    hit_mask = self.hits[hi,hj]>0
    hi = hi[hit_mask]
    hj = hj[hit_mask]

    # update_mask = random(size=len(hi))<0.8
    # self.grid[hi[update_mask], hj[update_mask]] = True
    self.grid[hi, hj] = True

