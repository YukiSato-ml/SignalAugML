# coding: utf-8
#chainer

import os
import sys
import chainer.initializers as I
from Core import Trainer
from Chains import classifier as C
from Chains import blocks
from Chains import chain

'''
PreActivate
layer 10
32-32x2x2-64x2x2-128x2x2-256x2x2-11
conv 1 + 2 x 4 x 2
fully 1
max pooling
ReLU
650
'''
class ResidualNN(C.Classifier):
    layer_num = [32,32,64,128,256]
    def __init__(self,initializer, layer_num=layer_num):
        super().__init__(chain.Res8Chain(initializer,blocks.PreActivateBlock_addNoise, layer_num, initialBN=True))

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
