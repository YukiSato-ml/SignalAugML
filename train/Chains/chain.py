# coding: utf-8
import chainer
import chainer.functions as F
import chainer.links as L
from . import gaussian as G
from . import padding_2d as P
from . import blocks
from . import util as U

class Res4Chain(chainer.Chain):
    def __init__(self, initializer, block, ln, initialBN=False):
        self.initialBatchNorm = initialBN
        self.count = U.ParameterCount()
        super(Res4Chain, self).__init__()
        with self.init_scope():
            self.conv = L.Convolution2D(1,ln[0], ksize=7, stride=1, pad=3, initialW=initializer)
            self.count.add(self.conv)
            self.bnorm = L.BatchNormalization(ln[0])
            self.block1 = block(initializer, ln[0], ln[1], self.count)
            self.block2 = block(initializer, ln[1], ln[2], self.count)
            self.block3 = block(initializer, ln[2], ln[3], self.count)
            self.block4 = block(initializer, ln[3], ln[4], self.count)
            self.line = L.Linear(ln[4], 11, initialW=initializer)  # n_units -> n_units
            self.count.add(self.line)
        self.counter = 0

    def __call__(self, x, test=False):
        h0 = self.conv(x) #43,128
        if(self.initialBatchNorm):
            h0 = self.bnorm(h0, finetune=test)
        else:
            h0 = F.relu(h0)
        h1 = self.block1(h0,test=test)
        h1 = F.max_pooling_2d(h1, ksize=3, stride=3)#15,43
        h2 = self.block2(h1,test=test)
        h2 = F.max_pooling_2d(h2, ksize=3, stride=3)#5,15
        h3 = self.block3(h2,test=test)
        h3 = F.max_pooling_2d(h3, ksize=3, stride=3)#3,5
        h4 = self.block4(h3,test=test)
        h4 = F.max_pooling_2d(h4, ksize=(3,5), stride=(3,5))#1,1
        y = self.line(h4)

        if not test:
            self.counter += 1
        return y

    def get_count(self):
        return self.counter

class Res4Chain2(chainer.Chain):
    def __init__(self, initializer, block, ln, initialBN=False):
        self.initialBatchNorm = initialBN
        self.count = U.ParameterCount()
        super(Res4Chain2, self).__init__()
        with self.init_scope():
            self.conv = L.Convolution2D(1,ln[0], ksize=3, stride=1, pad=1, initialW=initializer)
            self.count.add(self.conv)
            self.bnorm = L.BatchNormalization(ln[0])
            self.block1 = block(initializer, ln[0], ln[1], self.count)
            self.block2 = block(initializer, ln[1], ln[2], self.count)
            self.block3 = block(initializer, ln[2], ln[3], self.count)
            self.block4 = block(initializer, ln[3], ln[4], self.count)
            self.line = L.Linear(ln[4], 11, initialW=initializer)  # n_units -> n_units
            self.count.add(self.line)
        self.counter = 0

    def __call__(self, x, test=False):
        h0 = self.conv(x) #43,128
        if(self.initialBatchNorm):
            h0 = self.bnorm(h0, finetune=test)
        else:
            h0 = F.relu(h0)
        h1 = self.block1(h0,test=test)
        h1 = F.max_pooling_2d(h1, ksize=3, stride=3)#15,43
        h2 = self.block2(h1,test=test)
        h2 = F.max_pooling_2d(h2, ksize=3, stride=3)#5,15
        h3 = self.block3(h2,test=test)
        h3 = F.max_pooling_2d(h3, ksize=3, stride=3)#3,5
        h4 = self.block4(h3,test=test)
        h4 = F.max_pooling_2d(h4, ksize=(3,5), stride=(3,5))#1,1
        y = self.line(h4)

        if not test:
            self.counter += 1
        return y

    def get_count(self):
        return self.counter

class Res4Chain3(chainer.Chain):
    def __init__(self, initializer, block, ln, initialBN=False):
        self.initialBatchNorm = initialBN
        self.count = U.ParameterCount()
        super(Res4Chain2, self).__init__()
        with self.init_scope():
            self.conv = L.Convolution2D(1,ln[0], ksize=3, stride=1, pad=1, initialW=initializer)
            self.count.add(self.conv)
            self.bnorm = L.BatchNormalization(ln[0])
            self.block1 = block(initializer, ln[0], ln[1], self.count)
            self.block2 = block(initializer, ln[1], ln[2], self.count)
            self.block3 = block(initializer, ln[2], ln[3], self.count)
            self.block4 = block(initializer, ln[3], ln[4], self.count)
            self.line = L.Linear(ln[4], 11, initialW=initializer)  # n_units -> n_units
            self.count.add(self.line)
        self.counter = 0

    def __call__(self, x, test=False):
        h0 = self.conv(x) #43,512
        if(self.initialBatchNorm):
            h0 = self.bnorm(h0, finetune=test)
        else:
            h0 = F.relu(h0)
        h1 = self.block1(h0,test=test)
        h1 = F.max_pooling_2d(h1, ksize=(3,4), stride=(3,4))#15,128
        h2 = self.block2(h1,test=test)
        h2 = F.max_pooling_2d(h2, ksize=(3,4), stride=(3,4))#5,32
        h3 = self.block3(h2,test=test)
        h3 = F.max_pooling_2d(h3, ksize=(3,4), stride=(3,4))#3,8
        h4 = self.block4(h3,test=test)
        h4 = F.max_pooling_2d(h4, ksize=(3,8), stride=(3,8))#1,1
        y = self.line(h4)

        if not test:
            self.counter += 1
        return y

    def get_count(self):
        return self.counter

class ResPreactivate4Chain_limitedNoises(chainer.Chain):
    def __init__(self, initializer, ln, initialBN=False, limit=0):
        self.initialBatchNorm = initialBN
        self.count = U.ParameterCount()
        super(ResPreactivate4Chain_limitedNoises, self).__init__()
        with self.init_scope():
            self.conv = L.Convolution2D(1,ln[0], ksize=3, stride=1, pad=1, initialW=initializer)
            self.count.add(self.conv)
            self.bnorm = L.BatchNormalization(ln[0])
            self.block1 = blocks.PreActivateBlock_limitedNoise(initializer, ln[0], ln[1], self.count, limit)
            self.block2 = blocks.PreActivateBlock_limitedNoise(initializer, ln[1], ln[2], self.count, limit)
            self.block3 = blocks.PreActivateBlock_limitedNoise(initializer, ln[2], ln[3], self.count, limit)
            self.block4 = blocks.PreActivateBlock_limitedNoise(initializer, ln[3], ln[4], self.count, limit)
            self.line = L.Linear(ln[4], 11, initialW=initializer)  # n_units -> n_units
            self.count.add(self.line)
        self.counter = 0

    def __call__(self, x, test=False):
        h0 = self.conv(x) #43,128
        if(self.initialBatchNorm):
            h0 = self.bnorm(h0, finetune=test)
        else:
            h0 = F.relu(h0)
        h1 = self.block1(h0,self.counter,test=test)
        h1 = F.max_pooling_2d(h1, ksize=3, stride=3)#15,43
        h2 = self.block2(h1,self.counter,test=test)
        h2 = F.max_pooling_2d(h2, ksize=3, stride=3)#5,15
        h3 = self.block3(h2,self.counter,test=test)
        h3 = F.max_pooling_2d(h3, ksize=3, stride=3)#3,5
        h4 = self.block4(h3,self.counter,test=test)
        h4 = F.max_pooling_2d(h4, ksize=(3,5), stride=(3,5))#1,1
        y = self.line(h4)

        if not test:
            self.counter += 1
        return y

    def get_count(self):
        return self.counter

class Res5Chain(chainer.Chain):
    def __init__(self, initializer, block, ln, initialBN=False):
        self.initialBatchNorm = initialBN
        self.count = U.ParameterCount()
        super(Res5Chain, self).__init__()
        with self.init_scope():
            self.conv = L.Convolution2D(1,ln[0], ksize=7, stride=1, pad=3, initialW=initializer)
            self.count.add(self.conv)
            self.bnorm = L.BatchNormalization(ln[0])
            self.block1 = block(initializer, ln[0], ln[1], self.count)
            self.block2 = block(initializer, ln[1], ln[2], self.count)
            self.block3 = block(initializer, ln[2], ln[3], self.count)
            self.block4 = block(initializer, ln[3], ln[4], self.count)
            self.block5 = block(initializer, ln[4], ln[5], self.count)
            self.line = L.Linear(ln[5], 11, initialW=initializer)
            self.count.add(self.line)
        self.counter = 0

    def __call__(self, x, test=False):
        h0 = self.conv(x)#43,128
        if(self.initialBatchNorm):
            h0 = self.bnorm(h0, finetune=test)
        else:
            h0 = F.relu(h0)
        h1 = self.block1(h0,test=test)
        h1 = F.max_pooling_2d(h1, ksize=(1,3), stride=(1,3)) #43,43
        h2 = self.block2(h1,test=test)
        h2 = F.max_pooling_2d(h2, ksize=3, stride=3) #15, 15
        h3 = self.block3(h2,test=test)
        h3 = F.max_pooling_2d(h3, ksize=3, stride=3) #5, 5
        h4 = self.block4(h3,test=test)
        h4 = F.max_pooling_2d(h4, ksize=3, stride=3) #2, 2
        h5 = self.block5(h4,test=test)
        h5 = F.max_pooling_2d(h5, ksize=2, stride=2) #1, 1
        y = self.line(h5)

        if not test:
            self.counter += 1
        return y

    def get_count(self):
        return self.counter

class Res8Chain(chainer.Chain):
    def __init__(self, initializer, block, ln, initialBN=False):
        self.initialBatchNorm = initialBN
        self.count = U.ParameterCount()
        super(Res8Chain, self).__init__()
        with self.init_scope():
            self.conv = L.Convolution2D(1,ln[0], ksize=3, stride=1, pad=1, initialW=initializer)
            self.count.add(self.conv)
            self.bnorm = L.BatchNormalization(ln[0])
            self.block1 = block(initializer, ln[0], ln[1], self.count)
            self.block2 = block(initializer, ln[1], ln[1], self.count)
            self.block3 = block(initializer, ln[1], ln[2], self.count)
            self.block4 = block(initializer, ln[2], ln[2], self.count)
            self.block5 = block(initializer, ln[2], ln[3], self.count)
            self.block6 = block(initializer, ln[3], ln[3], self.count)
            self.block7 = block(initializer, ln[3], ln[4], self.count)
            self.block8 = block(initializer, ln[4], ln[4], self.count)
            self.line = L.Linear(ln[4], 11, initialW=initializer)
            self.count.add(self.line)
        self.counter = 0

    def __call__(self, x, test=False):
        h0 = self.conv(x)#43,128
        if(self.initialBatchNorm):
            h0 = self.bnorm(h0, finetune=test)
        else:
            h0 = F.relu(h0)
        h1 = self.block1(h0,test=test)
        h2 = self.block2(h1,test=test)
        h2 = F.max_pooling_2d(h2, ksize=3, stride=3)#15,43
        h3 = self.block3(h2,test=test)
        h4 = self.block4(h3,test=test)
        h4 = F.max_pooling_2d(h4, ksize=3, stride=3)#5,15
        h5 = self.block5(h4,test=test)
        h6 = self.block6(h5,test=test)
        h6 = F.max_pooling_2d(h6, ksize=3, stride=3)#3,5
        h7 = self.block7(h6,test=test)
        h8 = self.block8(h7,test=test)
        h8 = F.max_pooling_2d(h8, ksize=(3,5), stride=(3,5))#1,1
        y = self.line(h8)

        if not test:
            self.counter += 1
        return y

    def get_count(self):
        return self.counter

class Res10Chain(chainer.Chain):
    def __init__(self, initializer, block, ln, initialBN=False):
        self.initialBatchNorm = initialBN
        self.count = U.ParameterCount()
        super(Res10Chain, self).__init__()
        with self.init_scope():
            self.conv = L.Convolution2D(1,ln[0], ksize=7, stride=1, pad=3, initialW=initializer)
            self.count.add(self.conv)
            self.bnorm = L.BatchNormalization(ln[0])
            self.block1_1 = block(initializer, ln[0], ln[1], self.count)
            self.block1_2 = block(initializer, ln[1], ln[1], self.count)
            self.block2_1 = block(initializer, ln[1], ln[2], self.count)
            self.block2_2 = block(initializer, ln[2], ln[2], self.count)
            self.block3_1 = block(initializer, ln[2], ln[3], self.count)
            self.block3_2 = block(initializer, ln[3], ln[3], self.count)
            self.block4_1 = block(initializer, ln[3], ln[4], self.count)
            self.block4_2 = block(initializer, ln[4], ln[4], self.count)
            self.block5_1 = block(initializer, ln[4], ln[5], self.count)
            self.block5_2 = block(initializer, ln[5], ln[5], self.count)
            self.line = L.Linear(ln[5], 11, initialW=initializer)
            self.count.add(self.line)
        self.counter = 0

    def __call__(self, x, test=False):
        h0 = self.conv(x)
        if(self.initialBatchNorm):
            h0 = self.bnorm(h0, finetune=test)
        else:
            h0 = F.relu(h0) #43,128
        h1 = self.block1_1(h0,test=test)
        h1 = self.block1_2(h1,test=test)
        h1 = F.max_pooling_2d(h1, ksize=(1,3), stride=(1,3)) #64,43,43
        h2 = self.block2_1(h1,test=test)
        h2 = self.block2_2(h2,test=test)
        h2 = F.max_pooling_2d(h2, ksize=3, stride=3) #128, 15, 15
        h3 = self.block3_1(h2,test=test)
        h3 = self.block3_2(h3,test=test)
        h3 = F.max_pooling_2d(h3, ksize=3, stride=3) #256, 5, 5
        h4 = self.block4_1(h3,test=test)
        h4 = self.block4_2(h4,test=test)
        h4 = F.max_pooling_2d(h4, ksize=3, stride=3) #512, 2, 2
        h5 = self.block5_1(h4,test=test)
        h5 = self.block5_2(h5,test=test)
        h5 = F.max_pooling_2d(h5, ksize=2, stride=2) #512, 1, 1
        y = self.line(h5)

        if not test:
            self.counter += 1
        return y

    def get_count(self):
        return self.counter

class Res12Chain(chainer.Chain):
    def __init__(self, initializer, block, ln, initialBN=False):
        self.initialBatchNorm = initialBN
        self.count = U.ParameterCount()
        super(Res12Chain, self).__init__()
        with self.init_scope():
            self.conv = L.Convolution2D(1,ln[0], ksize=3, stride=1, pad=1, initialW=initializer)
            self.count.add(self.conv)
            self.bnorm = L.BatchNormalization(ln[0])
            self.block1_1 = block(initializer, ln[0], ln[1], self.count)
            self.block1_2 = block(initializer, ln[1], ln[1], self.count)
            self.block1_3 = block(initializer, ln[1], ln[1], self.count)

            self.block2_1 = block(initializer, ln[1], ln[2], self.count)
            self.block2_2 = block(initializer, ln[2], ln[2], self.count)
            self.block2_3 = block(initializer, ln[2], ln[2], self.count)

            self.block3_1 = block(initializer, ln[2], ln[3], self.count)
            self.block3_2 = block(initializer, ln[3], ln[3], self.count)
            self.block3_3 = block(initializer, ln[3], ln[3], self.count)

            self.block4_1 = block(initializer, ln[3], ln[4], self.count)
            self.block4_2 = block(initializer, ln[4], ln[4], self.count)
            self.block4_3 = block(initializer, ln[4], ln[4], self.count)

            self.line = L.Linear(ln[4], 11, initialW=initializer)
            self.count.add(self.line)
        self.counter = 0

    def __call__(self, x, test=False):
        h00 = self.conv(x)#43,128
        if(self.initialBatchNorm):
            h00 = self.bnorm(h00, finetune=test)
        else:
            h00 = F.relu(h00)
        h01 = self.block1_1(h00,test=test)
        h02 = self.block1_2(h01,test=test)
        h03 = self.block1_3(h02,test=test)
        h04 = F.max_pooling_2d(h03, ksize=3, stride=3)#15,43

        h05 = self.block2_1(h04,test=test)
        h06 = self.block2_2(h05,test=test)
        h07 = self.block2_3(h06,test=test)
        h08 = F.max_pooling_2d(h07, ksize=3, stride=3)#5,15

        h09 = self.block3_1(h08,test=test)
        h10 = self.block3_2(h09,test=test)
        h11 = self.block3_3(h10,test=test)
        h12 = F.max_pooling_2d(h11, ksize=3, stride=3)#3,5

        h13 = self.block4_1(h12,test=test)
        h14 = self.block4_2(h13,test=test)
        h15 = self.block4_3(h14,test=test)
        h16 = F.max_pooling_2d(h15, ksize=(3,5), stride=(3,5))#1,1

        y = self.line(h16)

        if not test:
            self.counter += 1
        return y

    def get_count(self):
        return self.counter

class Res16Chain(chainer.Chain):
    def __init__(self, initializer, block, ln, initialBN=False):
        self.initialBatchNorm = initialBN
        self.count = U.ParameterCount()
        super(Res16Chain, self).__init__()
        with self.init_scope():
            self.conv = L.Convolution2D(1,ln[0], ksize=7, stride=1, pad=3, initialW=initializer)
            self.count.add(self.conv)
            self.bnorm = L.BatchNormalization(ln[0])
            self.block1_1 = block(initializer, ln[0], ln[1], self.count)
            self.block1_2 = block(initializer, ln[1], ln[1], self.count)
            self.block1_3 = block(initializer, ln[1], ln[1], self.count)
            self.block1_4 = block(initializer, ln[1], ln[1], self.count)
            self.block2_1 = block(initializer, ln[1], ln[2], self.count)
            self.block2_2 = block(initializer, ln[2], ln[2], self.count)
            self.block2_3 = block(initializer, ln[2], ln[2], self.count)
            self.block2_4 = block(initializer, ln[2], ln[2], self.count)
            self.block3_1 = block(initializer, ln[2], ln[3], self.count)
            self.block3_2 = block(initializer, ln[3], ln[3], self.count)
            self.block3_3 = block(initializer, ln[3], ln[3], self.count)
            self.block3_4 = block(initializer, ln[3], ln[3], self.count)
            self.block4_1 = block(initializer, ln[3], ln[4], self.count)
            self.block4_2 = block(initializer, ln[4], ln[4], self.count)
            self.block4_3 = block(initializer, ln[4], ln[4], self.count)
            self.block4_4 = block(initializer, ln[4], ln[4], self.count)
            self.line = L.Linear(ln[4], 11, initialW=initializer)
            self.count.add(self.line)
        self.counter = 0

    def __call__(self, x, test=False):
        h00 = self.conv(x)#43,128
        if(self.initialBatchNorm):
            h00 = self.bnorm(h00, finetune=test)
        else:
            h00 = F.relu(h00)
        h01 = self.block1_1(h00,test=test)
        h02 = self.block1_2(h01,test=test)
        h03 = self.block1_3(h02,test=test)
        h04 = self.block1_4(h03,test=test)
        h04 = F.max_pooling_2d(h04, ksize=3, stride=3)#15,43

        h05 = self.block2_1(h04,test=test)
        h06 = self.block2_2(h05,test=test)
        h07 = self.block2_3(h06,test=test)
        h08 = self.block2_4(h07,test=test)
        h08 = F.max_pooling_2d(h08, ksize=3, stride=3)#5,15

        h09 = self.block3_1(h08,test=test)
        h10 = self.block3_2(h09,test=test)
        h11 = self.block3_3(h10,test=test)
        h12 = self.block3_4(h11,test=test)
        h12 = F.max_pooling_2d(h12, ksize=3, stride=3)#3,5

        h13 = self.block4_1(h12,test=test)
        h14 = self.block4_2(h13,test=test)
        h15 = self.block4_3(h14,test=test)
        h16 = self.block4_4(h15,test=test)
        h16 = F.max_pooling_2d(h16, ksize=(3,5), stride=(3,5))#1,1

        y = self.line(h16)

        if not test:
            self.counter += 1
        return y

    def get_count(self):
        return self.counter

class Res20Chain(chainer.Chain):
    def __init__(self, initializer, block, ln, initialBN=False):
        self.initialBatchNorm = initialBN
        self.count = U.ParameterCount()
        super(Res20Chain, self).__init__()
        with self.init_scope():
            self.conv = L.Convolution2D(1,ln[0], ksize=7, stride=1, pad=3, initialW=initializer)
            self.count.add(self.conv)
            self.bnorm = L.BatchNormalization(ln[0])
            self.block1_1 = block(initializer, ln[0], ln[1], self.count)
            self.block1_2 = block(initializer, ln[1], ln[1], self.count)
            self.block1_3 = block(initializer, ln[1], ln[1], self.count)
            self.block1_4 = block(initializer, ln[1], ln[1], self.count)
            self.block2_1 = block(initializer, ln[1], ln[2], self.count)
            self.block2_2 = block(initializer, ln[2], ln[2], self.count)
            self.block2_3 = block(initializer, ln[2], ln[2], self.count)
            self.block2_4 = block(initializer, ln[2], ln[2], self.count)
            self.block3_1 = block(initializer, ln[2], ln[3], self.count)
            self.block3_2 = block(initializer, ln[3], ln[3], self.count)
            self.block3_3 = block(initializer, ln[3], ln[3], self.count)
            self.block3_4 = block(initializer, ln[3], ln[3], self.count)
            self.block4_1 = block(initializer, ln[3], ln[4], self.count)
            self.block4_2 = block(initializer, ln[4], ln[4], self.count)
            self.block4_3 = block(initializer, ln[4], ln[4], self.count)
            self.block4_4 = block(initializer, ln[4], ln[4], self.count)
            self.block5_1 = block(initializer, ln[4], ln[5], self.count)
            self.block5_2 = block(initializer, ln[5], ln[5], self.count)
            self.block5_3 = block(initializer, ln[5], ln[5], self.count)
            self.block5_4 = block(initializer, ln[5], ln[5], self.count)
            self.line = L.Linear(ln[5], 11, initialW=initializer)
            self.count.add(self.line)
        self.counter = 0

    def __call__(self, x, test=False):
        h00 = self.conv(x)#43,128
        if(self.initialBatchNorm):
            h00 = self.bnorm(h00, finetune=test)
        else:
            h00 = F.relu(h00) #32,43,128

        h01 = self.block1_1(h00,test=test)
        h02 = self.block1_2(h01,test=test)
        h03 = self.block1_3(h02,test=test)
        h04 = self.block1_4(h03,test=test)
        h04 = F.max_pooling_2d(h04, ksize=(1,3), stride=(1,3)) #43,43

        h05 = self.block2_1(h04,test=test)
        h06 = self.block2_2(h05,test=test)
        h07 = self.block2_3(h06,test=test)
        h08 = self.block2_4(h07,test=test)
        h08 = F.max_pooling_2d(h08, ksize=3, stride=3) #15, 15

        h09 = self.block3_1(h08,test=test)
        h10 = self.block3_2(h09,test=test)
        h11 = self.block3_3(h10,test=test)
        h12 = self.block3_4(h11,test=test)
        h12 = F.max_pooling_2d(h12, ksize=3, stride=3) #5, 5

        h13 = self.block4_1(h12,test=test)
        h14 = self.block4_2(h13,test=test)
        h15 = self.block4_3(h14,test=test)
        h16 = self.block4_4(h15,test=test)
        h16 = F.max_pooling_2d(h16, ksize=3, stride=3) #2, 2

        h17 = self.block5_1(h16,test=test)
        h18 = self.block5_2(h17,test=test)
        h19 = self.block5_3(h18,test=test)
        h20 = self.block5_4(h19,test=test)
        h20 = F.max_pooling_2d(h20, ksize=2, stride=2) #1, 1

        y = self.line(h20)

        if not test:
            self.counter += 1
        return y

    def get_count(self):
        return self.counter
'''
class Res4Chain2(chainer.Chain):
    def __init__(self, initializer, block, ln, initialBN=False):
        self.initialBatchNorm = initialBN
        self.count = U.ParameterCount()
        self.pool = F.max_pooling_2d
        super(Res4Chain2, self).__init__()
        with self.init_scope():
            self.conv = L.Convolution2D(1,ln[0], ksize=7, stride=1, pad=3, initialW=initializer)
            self.count.add(self.conv)
            self.bnorm = L.BatchNormalization(ln[0])
            self.block1 = block(initializer, ln[0], ln[1], self.count)
            self.block2 = block(initializer, ln[1], ln[2], self.count)
            self.block3 = block(initializer, ln[2], ln[3], self.count)
            self.block4 = block(initializer, ln[3], ln[4], self.count)
            self.line = L.Linear(ln[4], 11, initialW=initializer)
            self.count.add(self.line)
        self.counter = 0

    def __call__(self, x, test=False):
        h0 = self.conv(x)#43,128
        if(self.initialBatchNorm):
            h0 = self.bnorm(h0, finetune=test)
        else:
            h0 = F.relu(h0)
        h0 = self.pool(h0, ksize=(1,3), stride=(1,3)) #43,43
        h1 = self.block1(h0,test=test)
        h1 = self.pool(h1, ksize=3, stride=3) #15, 15
        h2 = self.block2(h1,test=test)
        h2 = self.pool(h2, ksize=3, stride=3) #5, 5
        h3 = self.block3(h2,test=test)
        h3 = self.pool(h3, ksize=3, stride=3) #2, 2
        h4 = self.block4(h3,test=test)
        h4 = self.pool(h4, ksize=2) #1, 1
        y = self.line(h4)

        if not test:
            self.counter += 1
        return y


    def get_count(self):
        return self.counter
'''

class Res5ChainAlpha(chainer.Chain):
    def __init__(self, initializer, block, ln, initialBN=False):
        self.initialBatchNorm = initialBN
        self.count = U.ParameterCount()
        self.pool = F.max_pooling_2d
        super(Res5ChainAlpha, self).__init__()
        with self.init_scope():
            self.conv = L.Convolution2D(1,ln[0], ksize=7, stride=1, pad=3, initialW=initializer)
            self.count.add(self.conv)
            self.bnorm = L.BatchNormalization(ln[0])
            self.block1 = block(initializer, ln[0], ln[1], self.count)
            self.block2 = block(initializer, ln[1], ln[2], self.count)
            self.block3 = block(initializer, ln[2], ln[3], self.count)
            self.block4 = block(initializer, ln[3], ln[4], self.count)
            self.block5 = block(initializer, ln[4], ln[5], self.count)
            self.line = L.Linear(ln[5], 11, initialW=initializer)
            self.count.add(self.line)
        self.counter = 0

    def __call__(self, x, test=False):
        h0 = self.conv(x)#43,128
        if(self.initialBatchNorm):
            h0 = self.bnorm(h0, finetune=test)
        else:
            h0 = F.relu(h0)
        h0 = P.padding_2d(h0, 2, 0)#47,132
        h0 = self.pool(h0, ksize=3, stride=3) #16, 44
        h1 = self.block1(h0,test=test)
        h1 = P.padding_2d(h1, 2, 0)#20,48
        h1 = self.pool(h1, ksize=3, stride=3) #7,16
        h2 = self.block2(h1,test=test)
        h2 = P.padding_2d(h2, 2, 0)#11,20
        h2 = self.pool(h2, ksize=3, stride=3) #4,7
        h3 = self.block3(h2,test=test)
        h3 = P.padding_2d(h3, 2, 0)#8,11
        h3 = self.pool(h3, ksize=3, stride=3) #3,4
        h4 = self.block4(h3,test=test)
        h4 = P.padding_2d(h4, 2, 0)#7,8
        h4 = self.pool(h4, ksize=3, stride=3) #3,3
        h5 = self.block5(h4,test=test)
        h5 = self.pool(h5, ksize=3, stride=3) #1, 1
        y = self.line(h5)

        if not test:
            self.counter += 1
        return y

    def get_count(self):
        return self.counter

class Res4ChainAlpha(chainer.Chain):
    def __init__(self, initializer, block, ln, initialBN=False):
        self.initialBatchNorm = initialBN
        self.count = U.ParameterCount()
        self.pool = F.max_pooling_2d
        super(Res4ChainAlpha, self).__init__()
        with self.init_scope():
            self.conv = L.Convolution2D(1,ln[0], ksize=7, stride=1, pad=3, initialW=initializer)
            self.count.add(self.conv)
            self.bnorm = L.BatchNormalization(ln[0])
            self.block1 = block(initializer, ln[0], ln[1], self.count)
            self.block2 = block(initializer, ln[1], ln[2], self.count)
            self.block3 = block(initializer, ln[2], ln[3], self.count)
            self.block4 = block(initializer, ln[3], ln[4], self.count)
            self.line = L.Linear(ln[4], 11, initialW=initializer)
            self.count.add(self.line)
        self.counter = 0

    def __call__(self, x, test=False):
        h0 = self.conv(x)#43,128
        if(self.initialBatchNorm):
            h0 = self.bnorm(h0, finetune=test)
        else:
            h0 = F.relu(h0)
        h0 = P.padding_2d(h0, 1, 0)#45,130
        h0 = self.pool(h0, ksize=3, stride=3) #15, 44
        h1 = self.block1(h0,test=test)
        h1 = P.padding_2d(h1, 1, 0)#17, 46
        h1 = self.pool(h1, ksize=3, stride=3) #6,16
        h2 = self.block2(h1,test=test)
        h2 = P.padding_2d(h2, 1, 0)#8,18
        h2 = self.pool(h2, ksize=3, stride=3) #3, 6
        h3 = self.block3(h2,test=test)
        h3 = P.padding_2d(h3, 1, 0)#5,8
        h3 = self.pool(h3, ksize=3, stride=3) #2,3
        h4 = self.block4(h3,test=test)
        h4 = self.pool(h4, ksize=3, stride=3) #1,1
        y = self.line(h4)

        if not test:
            self.counter += 1
        return y

    def get_count(self):
        return self.counter

class Res3Chain(chainer.Chain):
    def __init__(self, initializer, block, ln, initialBN=False, rand=False):
        self.initialBatchNorm = initialBN
        self.count = U.ParameterCount()
        self.pool = F.average_pooling_2d
        self.rand = rand
        super(Res3Chain, self).__init__()
        with self.init_scope():
            self.conv = L.Convolution2D(1,ln[0], ksize=7, stride=1, pad=3, initialW=initializer)
            self.count.add(self.conv)
            self.bnorm = L.BatchNormalization(ln[0])
            self.block1 = block(initializer, ln[0], ln[1], self.count)
            self.block2 = block(initializer, ln[1], ln[2], self.count)
            self.block3 = block(initializer, ln[2], ln[3], self.count)
            self.line = L.Linear(ln[3], 11, initialW=initializer)  # n_units -> n_units
            self.count.add(self.line)
        self.counter = 0

    def __call__(self, x, test=False, ):
        if(self.rand):
            if(U.rand_flag()):
                x = U.cut_out(x)
        h0 = self.conv(x) #43,128
        if(self.initialBatchNorm):
            h0 = self.bnorm(h0, finetune=test)
        else:
            h0 = F.relu(h0)
        h0 = self.pool(h0, ksize=3, stride=3)#14,42
        print(h0.data.shape)
        h1 = self.block1(h0,test=test)
        h1 = self.pool(h1, ksize=3, stride=3)#5,15
        h2 = self.block2(h1,test=test)
        h2 = self.pool(h2, ksize=3, stride=3)#3,5
        h3 = self.block3(h2,test=test)

        h3 = self.pool(h3, ksize=(3,5))#1,1
        y = self.line(h3)

        if not test:
            self.counter += 1
        return y

    def get_count(self):
        return self.counter

class Res6Chain(chainer.Chain):
    def __init__(self, initializer, block, ln, initialBN=False):
        self.initialBatchNorm = initialBN
        self.count = U.ParameterCount()
        self.pool = F.max_pooling_2d
        super(Res6Chain, self).__init__()
        with self.init_scope():
            self.conv = L.Convolution2D(1,ln[0], ksize=7, stride=1, pad=3, initialW=initializer)
            self.count.add(self.conv)
            self.bnorm = L.BatchNormalization(ln[0])
            self.block1_1 = block(initializer, ln[0], ln[1], self.count)
            self.block1_2 = block(initializer, ln[1], ln[1], self.count)
            self.block2_1 = block(initializer, ln[1], ln[2], self.count)
            self.block2_2 = block(initializer, ln[2], ln[2], self.count)
            self.block3_1 = block(initializer, ln[2], ln[3], self.count)
            self.block3_2 = block(initializer, ln[3], ln[3], self.count)
            self.line = L.Linear(ln[3], 11, initialW=initializer)  # n_units -> n_units
            self.count.add(self.line)
        self.counter = 0

    def __call__(self, x, test=False):
        h0 = self.conv(x) #43,128
        if(self.initialBatchNorm):
            h0 = self.bnorm(h0, finetune=test)
        else:
            h0 = F.relu(h0)
        h0 = self.pool(h0, ksize=3, stride=3)#15,43
        h1 = self.block1_1(h0,test=test)
        h2 = self.block1_2(h1,test=test)
        h2 = self.pool(h2, ksize=3, stride=3)#5,15
        h3 = self.block2_1(h2,test=test)
        h4 = self.block2_2(h3,test=test)
        h4 = self.pool(h4, ksize=3, stride=3)#3,5
        h5 = self.block3_1(h4,test=test)
        h6 = self.block3_2(h5,test=test)
        h6 = self.pool(h6, ksize=(3,5), stride=(3,5))#1,1
        y = self.line(h6)

        if not test:
            self.counter += 1
        return y

    def get_count(self):
        return self.counter

class Res5ChainBata(chainer.Chain):
    def __init__(self, initializer, block, ln, initialBN=False, rand=False):
        self.initialBatchNorm = initialBN
        self.count = U.ParameterCount()
        self.pool = F.average_pooling_2d
        self.rand = rand
        super(Res5ChainBata, self).__init__()
        with self.init_scope():
            self.conv = L.Convolution2D(1,ln[0], ksize=7, stride=1, pad=3, initialW=initializer)
            self.count.add(self.conv)
            self.bnorm = L.BatchNormalization(ln[0])
            self.block1 = block(initializer, ln[0], ln[1], self.count)
            self.block2 = block(initializer, ln[1], ln[2], self.count)
            self.block3 = block(initializer, ln[2], ln[3], self.count)
            self.block4 = block(initializer, ln[3], ln[4], self.count)
            self.block5 = block(initializer, ln[4], ln[5], self.count)
            self.line = L.Linear(ln[5], 11, initialW=initializer)  # n_units -> n_units
            self.count.add(self.line)
        self.counter = 0

    def __call__(self, x, test=False):
        x = F.upsampling_2d(x, ksize=(3,1)) #129,128
        print(x.data.shape)
        if(self.rand):
            if(U.rand_flag()):
                x = U.cut_out(x)
        h0 = self.conv(x)
        if(self.initialBatchNorm):
            h0 = self.bnorm(h0, finetune=test)
        else:
            h0 = F.relu(h0)
        h0 = self.pool(h0, ksize=2, stride=2)#64,64
        print(h0.data.shape)
        h1 = self.block1(h0,test=test)
        h1 = self.pool(h1, ksize=2, stride=2)#32,32
        h2 = self.block2(h1,test=test)
        h2 = self.pool(h2, ksize=2, stride=2)#16,16
        h3 = self.block3(h2,test=test)
        h3 = self.pool(h3, ksize=2, stride=2)#8,8
        h4 = self.block4(h3,test=test)
        h4 = self.pool(h4, ksize=2, stride=2)#4,4
        h5 = self.block5(h4,test=test)
        h5 = self.pool(h5, ksize=4, stride=4)#1,1
        y = self.line(h5)

        if not test:
            self.counter += 1
        return y

    def get_count(self):
        return self.counter

class PlainNet(chainer.Chain):
    def __init__(self, initializer, ln, initialBN=False):
        self.initialBatchNorm = initialBN
        self.count = U.ParameterCount()
        super(PlainNet, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(1,ln[0], ksize=3, stride=1, pad=2, initialW=initializer)
            self.conv2 = L.Convolution2D(ln[0],ln[0], ksize=3, stride=1, pad=2, initialW=initializer)
            self.count.add(self.conv1)
            self.count.add(self.conv2)
            self.conv3 = L.Convolution2D(ln[0],ln[1], ksize=3, stride=1, pad=2, initialW=initializer)
            self.conv4 = L.Convolution2D(ln[1],ln[1], ksize=3, stride=1, pad=2, initialW=initializer)
            self.count.add(self.conv3)
            self.count.add(self.conv4)
            self.conv5 = L.Convolution2D(ln[1],ln[2], ksize=3, stride=1, pad=2, initialW=initializer)
            self.conv6 = L.Convolution2D(ln[2],ln[2], ksize=3, stride=1, pad=2, initialW=initializer)
            self.count.add(self.conv5)
            self.count.add(self.conv6)
            self.conv7 = L.Convolution2D(ln[2],ln[3], ksize=3, stride=1, pad=2, initialW=initializer)
            self.conv8 = L.Convolution2D(ln[3],ln[3], ksize=3, stride=1, pad=2, initialW=initializer)
            self.count.add(self.conv7)
            self.count.add(self.conv8)
            self.line1 = L.Linear(ln[3], ln[4], initialW=initializer)
            self.line2 = L.Linear(ln[4], 11, initialW=initializer)  # n_units -> n_units
            self.count.add(self.line1)
            self.count.add(self.line2)
        self.counter = 0

    def __call__(self, x, test=False):
        h0 = F.relu(self.conv1(x)) #45,130
        h1 = F.relu(self.conv2(h0)) #47,132
        h1 = F.max_pooling_2d(h1, ksize=3, stride=3)#15,44
        h1 = F.dropout(h1,0.25)

        h2 = F.relu(self.conv3(h1)) #17,46
        h3 = F.relu(self.conv4(h2)) #19,48
        h3 = F.max_pooling_2d(h3, ksize=3, stride=3)#6,16
        h3 = F.dropout(h3,0.25)

        h4 = F.relu(self.conv5(h3)) #9,18
        h5 = F.relu(self.conv6(h4)) #11,20
        h5 = F.max_pooling_2d(h5, ksize=3, stride=3)#3,6
        h5 = F.dropout(h5,0.25)

        h6 = F.relu(self.conv7(h5)) #6,9
        h7 = F.relu(self.conv8(h6)) #8,11
        h7 = F.max_pooling_2d(h7, ksize=(8,11), stride=(7,10))#1,1

        h8 = F.relu(self.line1(h7))
        h8 = F.dropout(h8,0.5)

        y = self.line2(h8)

        if not test:
            self.counter += 1
        return y

    def get_count(self):
        return self.counter
