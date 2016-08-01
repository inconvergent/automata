#!/usr/bin/python3
# -*- coding: utf-8 -*-

from modules.helpers import plt_style
from time import sleep

GRID_SIZE = 1024 # power of two
ONE = 1.0/GRID_SIZE
ONEHALF = ONE*0.5

FRONT = [0,0,0,1]
BACK = [1,1,1,1]

THREADS = 512
ZONE_LEAP = 512

INFLUENCE_RAD = 3
CROWDED_LIMIT = 15

LEAP = 100

MS = 2.0
ALPHA = 0.9


def get_initial(num=10, shift=2):
  from numpy import zeros
  from numpy.random import randint
  init = zeros((GRID_SIZE, GRID_SIZE), 'bool')

  mid = int(GRID_SIZE/2)

  init[mid-shift:mid+shift,mid-shift:mid+shift] = True
  xx = randint(mid-shift,mid+shift, size=(num))
  yy = randint(mid-shift,mid+shift, size=(num))
  init[xx,yy] = False

  # init[mid-shift:mid+shift,mid] = True

  return init

@plt_style
def show(plt, automata):
  plt.clf()
  i, j = automata.grid.nonzero()
  x = i.astype('float')
  y = j.astype('float')

  # hi, hj = automata.hits.nonzero()
  # hit_mask = automata.hits[hi,hj]>0

  # hx = hi[hit_mask].astype('float')
  # hy = hj[hit_mask].astype('float')

  ci, cj = (automata.connected>1).nonzero()
  #
  # hx = hi[hit_mask].astype('float')
  # hy = hj[hit_mask].astype('float')
  #
  # cx = ci.astype('float')
  # cy = cj.astype('float')

  # print('sum', hit_count.sum(), len(hit_count))

  # massx = automata.massx[i, j]
  # massy = automata.massy[i, j]

  # from numpy import column_stack
  # print(column_stack((massx,massy)))
  # print(automata.neigh[i,j])
  # print(automata.hits)
  # print()

  x *= ONE
  y *= ONE
  # hx *= ONE
  # hy *= ONE
  # cx *= ONE
  # cy *= ONE

  # plt.quiver(
  #     x, y,
  #     massx, massy,
  #     units='width', scale=GRID_SIZE*0.25,
  #     headaxislength=0, headwidth=1,
  #     alpha=ALPHA, color='r'
  #     )
  # plt.plot(
  #     hx, hy,
  #     'bo',
  #     markersize=2*MS, alpha=ALPHA*0.5
  #     )
  # plt.plot(
  #     cx, cy,
  #     'go',
  #     markersize=2*MS, alpha=ALPHA*0.5
  #     )
  plt.plot(
      x, y,
      'k.',
      markersize=MS, alpha=ALPHA
      )



def main():
  from modules.automata import Automata
  from fn import Fn
  from matplotlib import animation
  import matplotlib.pyplot as plt

  fn = Fn(prefix='./res/', postfix='.png')

  fig = plt.figure('automata')

  A = Automata(
      GRID_SIZE,
      get_initial(num=50, shift=5),
      INFLUENCE_RAD,
      CROWDED_LIMIT,
      THREADS,
      )

  im = plt.imshow(A.grid, cmap='gray')
  plt.axis('off')
  plt.axes().set_aspect('equal', 'datalim')

  def init():
    im.set_data(A.grid)
    return im,

  def animate(i):
    im.set_data(A.grid)
    A.step()
    print(i, A.itt)
    if not i%LEAP:
      plt.pause(0.000000001)
      name = fn.name()
      plt.savefig(name, bbox_inches='tight', aspect='equal')
    return im,

  anim = animation.FuncAnimation(
      fig,
      animate,
      init_func=init,
      interval=0,
      )


  plt.show()


  # while True:
  #   try:
  #     A.step()
  #     # show(plt, A)
  #     show_imshow(plt, A)
  #     # plt.draw()
  #     if not A.itt % LEAP:
  #       print(A.itt)
  #       plt.pause(0.00001)
  #       name = fn.name()
  #       # plt.savefig(name, bbox_inches='tight', aspect='equal')
  #       plt.savefig(name)
  #   except KeyboardInterrupt:
  #     break



if __name__ == '__main__':
  main()

