#!/usr/bin/env python
# -*- coding: utf-8 -*-

from chainer import cuda
from chainer import configuration
from chainer import function
from chainer.utils import type_check

class BlackoutFunction(function.Function):
  def __init__(self, std):
    super().__init__()
    self.std = std

  def check_type_forward(self, in_types):
    type_check.expect(in_types.size() == 1)
    type_check.expect(in_types[0].dtype.kind == 'f')

  def forward(self, inputs):
    self.retain_inputs(())

    if self.std == 0.0:
      return inputs

    xp = cuda.get_array_module(*inputs)
    x = inputs[0]
    s = [x.shape[0], x.shape[1]] + [1] * (len(x.shape) - 2)
    p = xp.random.random(s).astype(xp.float32)
    x *= p > self.std

    return x,

  def backward(self, inputs, grad):
    return grad


def blackout(x, std):
  if configuration.config.train:
    return BlackoutFunction(std)(x)
  else:
    return x
