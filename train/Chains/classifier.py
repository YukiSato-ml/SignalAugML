# coding: utf-8
#chainer

from __future__ import print_function

import sys
import chainer
import numpy
#from chainer.cuda import cupy
from chainer import cuda
from chainer import serializers
from Chains import models as M
import chainer.functions as F

class Classifier(chainer.Chain):
    def __init__(self, predictor):
        super(Classifier,self).__init__(predictor=predictor)
        #print(predictor)

    def forword(self, x, t, batchsize):
        xp  = cuda.get_array_module(*x)
        tt = xp.zeros((batchsize))

        for i in range(batchsize):
            ss = xp.where(t[i] > 0.5)
            tt[i] = ss[0]

        x = xp.array(x, dtype=xp.float32)
        t = xp.array(t, dtype=xp.int32)
        tt = tt.astype(xp.int32)

        y = self.predictor(x)
        loss = F.sigmoid_cross_entropy(y, t)

        accuracy = F.accuracy(y, tt)

        return loss, accuracy

    def testing(self, x, n, maxpool):
        xp  = cuda.get_array_module(*x)
        label = 11
        x = xp.array(x, dtype=xp.float32)
        y = F.sigmoid(self.predictor(x, test=True))

        win = maxpool
        hop = int(win/2)
        task = int(n/hop)-1

        y_array = xp.array(y.data, dtype=xp.float64)

        if(n < win):
            tmp_mp = xp.zeros(1, label)
            tmp_mp[0] = y_array.max(axis = 0)

        else:
            tmp_mp = xp.zeros((task, label))
            for i in range(task):
                start = i*hop
                end = start + win
                y_win = y_array[start:end,:]

                tmp_mp[i] = y_win.max(axis = 0)

        prediction = tmp_mp.sum(axis = 0)
        #正規化
        prediction = (prediction - prediction.min())/(prediction.max() - prediction.min())

        return prediction

    def param_count(self):
        count = self.predictor.count
        return count.param_num, count.sum_param, count.param_list
