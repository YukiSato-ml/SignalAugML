# coding: utf-8
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda
import numpy as np

class ParameterCount():
    def __init__(self):
        self.param_list=[]
        self.param_num = 0
        self.sum_param = 0

    def add(self, var, nobias=False):
        w_shape = var.W.shape
        b_shape = var.b.shape if not nobias else None

        param_w = self.product(w_shape)
        param_b = self.product(b_shape) if not nobias else 0
        param = param_w + param_b

        self.sum_param += param
        self.param_list.append(param)
        self.param_num += 1

    def product(self, lis):
        y = 1
        for n in lis:
            y *= n

        return y

def zero_pad(x, inch, outch):
    xp = chainer.cuda.get_array_module(x.data)
    bn, ch, hh, ww = x.data.shape
    pad_ch = outch - inch
    pad_array = xp.zeros((bn, pad_ch, hh, ww), dtype=xp.float32)
    y = chainer.Variable(pad_array)
    return y

def rand_flag():
    flag = np.random.rand() * 10

    return flag < 5

def cut_out(x):
    xp = chainer.cuda.get_array_module(*x)
    bn, ch, hh, ww = x.shape
    cut_val = int(xp.random.rand() * (ww/2))+1
    cut_list = xp.array(xp.random.rand(cut_val) * ww, dtype='int')

    x[:,:,:,cut_list] = 0

    return x
