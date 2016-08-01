#!/usr/bin/python
# -*- coding: utf-8 -*-


def plt_style(func):
  def f(plt, *_):
    func(plt, *_)
    plt.axis('off')
    plt.axes().set_aspect('equal', 'datalim')
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.subplots_adjust(bottom=0.,left=0.,right=1.,top=1.)
  return f

def load_kernel(fn, name, subs={}):

  from pycuda.compiler import SourceModule

  with open(fn, 'r') as f:
    kernel = f.read()

  for k,v in subs.items():
    kernel = kernel.replace(k, str(v))

  mod = SourceModule(kernel)
  return mod.get_function(name)


# @plt_style
# def show(plt, automata):
#   plt.clf()
#   i, j = automata.grid.nonzero()
#   x = i.astype('float')
#   y = j.astype('float')
#
#   # hi, hj = automata.hits.nonzero()
#   # hit_mask = automata.hits[hi,hj]>0
#
#   # hx = hi[hit_mask].astype('float')
#   # hy = hj[hit_mask].astype('float')
#
#   ci, cj = (automata.connected>1).nonzero()
#   #
#   # hx = hi[hit_mask].astype('float')
#   # hy = hj[hit_mask].astype('float')
#   #
#   # cx = ci.astype('float')
#   # cy = cj.astype('float')
#
#   # print('sum', hit_count.sum(), len(hit_count))
#
#   # massx = automata.massx[i, j]
#   # massy = automata.massy[i, j]
#
#   # from numpy import column_stack
#   # print(column_stack((massx,massy)))
#   # print(automata.neigh[i,j])
#   # print(automata.hits)
#   # print()
#
#   x *= ONE
#   y *= ONE
#   # hx *= ONE
#   # hy *= ONE
#   # cx *= ONE
#   # cy *= ONE
#
#   # plt.quiver(
#   #     x, y,
#   #     massx, massy,
#   #     units='width', scale=GRID_SIZE*0.25,
#   #     headaxislength=0, headwidth=1,
#   #     alpha=ALPHA, color='r'
#   #     )
#   # plt.plot(
#   #     hx, hy,
#   #     'bo',
#   #     markersize=2*MS, alpha=ALPHA*0.5
#   #     )
#   # plt.plot(
#   #     cx, cy,
#   #     'go',
#   #     markersize=2*MS, alpha=ALPHA*0.5
#   #     )
#   plt.plot(
#       x, y,
#       'k.',
#       markersize=MS, alpha=ALPHA
#       )
