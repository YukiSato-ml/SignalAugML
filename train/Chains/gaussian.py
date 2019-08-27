#!/usr/bin/env python
# -*- coding: utf-8 -*-

from chainer import cuda
from chainer import configuration
from chainer import function
from chainer.utils import type_check

class GaussianNoiseFunction(function.Function):
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
    s = [x.shape[0]] + [1] * (len(x.shape) - 1)
    x *= xp.random.normal(1.0, self.std, s).astype(xp.float32)

    return x,

  def backward(self, inputs, grad):
    return grad


def gaussian_noise(x, std):
  if configuration.config.train:
    return GaussianNoiseFunction(std)(x)
  else:
    return x
