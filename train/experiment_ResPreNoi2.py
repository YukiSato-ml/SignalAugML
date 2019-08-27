# coding: utf-8
#chainer

import os
import sys
import chainer
import chainer.initializers as I
import chainer.functions as F
import chainer.links as L
from Core import Trainer
from Chains import classifier as C
from Chains import chain
from Chains import util as U
from Chains import gaussian as G

'''
PreActivate noise 2
layer 10
32-32x2-64x2-128x2-256x2-11
conv 1 + 2 x 4
fully 1
max pooling
ReLU
650
'''
class Block(chainer.Chain):
    def __init__(self, initializer, inch, outch, util):
        self.inch = inch
        self.outch = outch
        super(Block, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(inch,outch, ksize=3, stride=1, pad=1, initialW=initializer)
            util.add(self.conv1)
            self.conv2 = L.Convolution2D(outch,outch, ksize=3, stride=1, pad=1, initialW=initializer)
            util.add(self.conv2)
            self.bnorm1 = L.BatchNormalization(inch)
            self.bnorm2 = L.BatchNormalization(outch)
            self.bnorm3 = L.BatchNormalization(outch)

    def __call__(self, x, test=False):
        h0 = F.relu(self.bnorm1(x, finetune=test))
        h0= G.gaussian_noise(h0,0.8)
        h1 = F.relu(self.bnorm2(self.conv1(h0), finetune=test))
        h2 = self.bnorm3(self.conv2(h1), finetune=test)
        pad_x = F.concat((x, U.zero_pad(x, self.inch, self.outch)))
        h3 = h2+pad_x

        return h3

class ResidualNN(C.Classifier):
    layer_num = [32,32,64,128,256]
    def __init__(self,initializer, layer_num=layer_num):
        super().__init__(chain.Res4Chain2(initializer,Block, layer_num, initialBN=True))

export_dir = os.path.join(os.getcwd(),"save")
filename, ext= os.path.splitext(__file__)
dataset_dir = os.path.join(os.getcwd(),"../dataset")

train = Trainer(filename, export_dir, dataset_dir)
train.set_networks(ResidualNN, I.HeNormal)

train.model_init()
lr_update = [[0,200,350,500],[0.3,0.1,0.01,0.001]]

train.train_loop(lr_update=lr_update)

train.testing()

train.export()
