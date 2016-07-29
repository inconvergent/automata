#!/usr/bin/python3
# -*- coding: utf-8 -*-

from modules.helpers import plt_style

GRID_SIZE = 512
ONE = 1.0/GRID_SIZE
ONEHALF = ONE*0.5

FRONT = [0,0,0,1]
BACK = [1,1,1,1]

THREADS = 512
ZONE_LEAP = 512

INFLUENCE_RAD = 5

LEAP = 10

MS = 3.0
ALPHA = 0.5


def get_initial(num=500, shift=50):
  from numpy import zeros
  from numpy.random import randint
  init = zeros((GRID_SIZE, GRID_SIZE), 'bool')
  # init[:,int(GRID_SIZE/2)] = True

  xx = randint(GRID_SIZE*0.5-shift,GRID_SIZE*0.5+shift, size=(num))
  yy = randint(GRID_SIZE*0.5-shift,GRID_SIZE*0.5+shift, size=(num))

  init[xx,yy] = True
  return init

@plt_style
def show(plt, automata):
  plt.clf()
  i, j = automata.grid.nonzero()
  x = i.astype('float')
  y = j.astype('float')

  massx = automata.massx[i, j]
  massy = automata.massy[i, j]

  # from numpy import column_stack
  # print(column_stack((massx,massy)))
  # print(automata.neigh[i,j])
  # print()

  x *= ONE
  y *= ONE

  plt.quiver(
      x, y,
      massx, massy,
      units='width', scale=GRID_SIZE*0.25,
      headaxislength=0, headwidth=1,
      alpha=ALPHA, color='r'
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
      get_initial(),
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



if __name__ == '__main__':
  main()

