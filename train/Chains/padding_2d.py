#!/usr/bin/env python
# -*- coding: utf-8 -*-

from chainer import cuda
from chainer import configuration
from chainer import function as F
from chainer.utils import type_check

class Padding2D(F.Function):
    def __init__(self, size, val):
        super().__init__()
        self.size = size
        self.val = val

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        type_check.expect(in_types[0].dtype.kind == 'f')

    def forward(self, inputs):
        self.retain_inputs(())

        if self.size == 0:
            return inputs

        xp = cuda.get_array_module(*inputs)
        x = inputs[0]
        pad_size = self.size*2
        bt, ch, hh, ww = x.shape
        s = xp.ones((bt, ch, hh+pad_size, ww+pad_size), dtype='f')
        s *= self.val
        s[:, :, self.size:self.size*-1, self.size:self.size*-1] = x[:,:]

        return s,

    def backward(self, inputs, grad):
        if self.size == 0:
            return grad

        x = grad[0]
        y = x[:, :, self.size:self.size*-1, self.size:self.size*-1]

        return y,

def padding_2d(x, size, val):
    return Padding2D(size, val)(x)
