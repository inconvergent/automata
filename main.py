#!/usr/bin/python3
# -*- coding: utf-8 -*-

from modules.helpers import plt_style

GRID_SIZE = 512 # power of two
ONE = 1.0/GRID_SIZE
ONEHALF = ONE*0.5

FRONT = [0,0,0,1]
BACK = [1,1,1,1]

THREADS = 512
ZONE_LEAP = 512

INFLUENCE_RAD = 3

LEAP = 1

MS = 2.0
ALPHA = 0.5


def get_initial(num=10, shift=2):
  from numpy import zeros
  from numpy.random import randint
  init = zeros((GRID_SIZE, GRID_SIZE), 'bool')

  mid = int(GRID_SIZE/2)

  init[mid-shift:mid+shift,mid-shift:mid+shift] = True
  # xx = randint(mid-shift,mid+shift, size=(num))
  # yy = randint(mid-shift,mid+shift, size=(num))
  # init[xx,yy] = False

  # init[mid-shift:mid+shift,mid] = True

  return init

@plt_style
def show(plt, automata):
  plt.clf()
  i, j = automata.grid.nonzero()
  x = i.astype('float')
  y = j.astype('float')

  hi, hj = automata.hits.nonzero()
  hit_mask = automata.hits[hi,hj]>0

  hx = hi[hit_mask].astype('float')
  hy = hj[hit_mask].astype('float')

  # print('sum', hit_count.sum(), len(hit_count))

  massx = automata.massx[i, j]
  massy = automata.massy[i, j]

  # from numpy import column_stack
  # print(column_stack((massx,massy)))
  # print(automata.neigh[i,j])
  # print(automata.hits)
  # print()

  x *= ONE
  y *= ONE
  hx *= ONE
  hy *= ONE

  # plt.quiver(
  #     x, y,
  #     massx, massy,
  #     units='width', scale=GRID_SIZE*0.25,
  #     headaxislength=0, headwidth=1,
  #     alpha=ALPHA, color='r'
  #     )
  plt.plot(
      hx, hy,
      'bo',
      markersize=2*MS, alpha=ALPHA*0.5
      )
  plt.plot(
      x, y,
      'ko',
      markersize=MS, alpha=ALPHA
      )

  plt.draw()



def main():
  from modules.automata import Automata

  import matplotlib.pyplot as plt
  plt.ion()

  A = Automata(
      GRID_SIZE,
      get_initial(num=100, shift=4),
      INFLUENCE_RAD,
      THREADS,
      )

  while True:
    try:
      A.step()
      show(plt, A)
      if not A.itt % LEAP:
        print(A.itt)
        plt.pause(0.05)
    except KeyboardInterrupt:
      break

  plt.ioff()
  show(plt, A)
  plt.show()



if __name__ == '__main__':
  main()

