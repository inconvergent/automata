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

