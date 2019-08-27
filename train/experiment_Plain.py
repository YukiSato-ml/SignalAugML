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

'''
PreActivate
layer 10
32-32x2-64x2-128x2-256x2-11
conv 1 + 2 x 4
fully 1
max pooling
ReLU
650
'''
class ResidualNN(C.Classifier):
    layer_num = [32,64,128,256,1024]
    def __init__(self,initializer, layer_num=layer_num):
        super().__init__(chain.PlainNet(initializer, layer_num))

export_dir = os.path.join(os.getcwd(),"save")
filename, ext= os.path.splitext(__file__)
dataset_dir = os.path.join(os.getcwd(),"../dataset")

train = Trainer(filename, export_dir, dataset_dir)
train.set_networks(ResidualNN, I.HeNormal)

train.model_init()
lr_update = [[0,200,350,500],[0.01,0.005,0.001,0.0001]]

train.train_loop(lr_update=lr_update)

train.testing()

train.export()
