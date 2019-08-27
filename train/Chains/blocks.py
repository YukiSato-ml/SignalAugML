# coding: utf-8
import chainer
import chainer.functions as F
import chainer.links as L
from . import gaussian as G
from . import util as U

#Residual NN
class ResBlock(chainer.Chain):
    def __init__(self, initializer, inch, outch, util):
        self.inch = inch
        self.outch = outch
        super(ResBlock, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(inch,outch, ksize=3, stride=1, pad=1, initialW=initializer)
            util.add(self.conv1)
            self.conv2 = L.Convolution2D(outch,outch, ksize=3, stride=1, pad=1, initialW=initializer)
            util.add(self.conv2)

    def __call__(self, x, test=False):
        h1 = F.relu(self.conv1(x))
        h2 = F.relu(self.conv2(h1))

        pad_x = F.concat((x, U.zero_pad(x, self.inch, self.outch)))
        h3 = h2 + pad_x

        return h3

class ResBlock_addNoise(chainer.Chain):
    def __init__(self, initializer, inch, outch, util):
        self.inch = inch
        self.outch = outch
        super(ResBlock_addNoise, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(inch,outch, ksize=3, stride=1, pad=1, initialW=initializer)
            util.add(self.conv1)
            self.conv2 = L.Convolution2D(outch,outch, ksize=3, stride=1, pad=1, initialW=initializer)
            util.add(self.conv2)

    def __call__(self, x, test=False):
        h1 = F.relu(self.conv1(x))
        h2 = F.relu(self.conv2(h1))
        h2 = G.gaussian_noise(h2,0.5)
        pad_x = F.concat((x, U.zero_pad(x, self.inch, self.outch)))
        h3 = h2 + pad_x

        return h3

class ResBlock_addDO(chainer.Chain):
    def __init__(self, initializer, inch, outch, util):
        self.inch = inch
        self.outch = outch
        super(ResBlock_addDO, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(inch,outch, ksize=3, stride=1, pad=1, initialW=initializer)
            util.add(self.conv1)
            self.conv2 = L.Convolution2D(outch,outch, ksize=3, stride=1, pad=1, initialW=initializer)
            util.add(self.conv2)

    def __call__(self, x, test=False):
        h1 = self.conv1(x)
        h2 = F.relu(F.dropout(h1,0.2))
        h3 = F.relu(self.conv2(h2))
        pad_x = F.concat((x, U.zero_pad(x, self.inch, self.outch)))
        h4 = h3 + pad_x

        return h3

class SingleReluBlock(chainer.Chain):
    def __init__(self, initializer, inch, outch, util):
        self.inch = inch
        self.outch = outch
        super(SingleReluBlock, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(inch,outch, ksize=3, stride=1, pad=1, initialW=initializer)
            util.add(self.conv1)
            self.conv2 = L.Convolution2D(outch,outch, ksize=3, stride=1, pad=1, initialW=initializer)
            util.add(self.conv2)
            self.bnorm1 = L.BatchNormalization(inch)
            self.bnorm2 = L.BatchNormalization(outch)
            self.bnorm3 = L.BatchNormalization(outch)

    def __call__(self, x, test=False):
        h0 = self.bnorm1(x, finetune=test)
        h1 = F.relu(self.bnorm2(self.conv1(h0), finetune=test))
        h2 = self.bnorm3(self.conv2(h1), finetune=test)

        pad_x = F.concat((x, U.zero_pad(x, self.inch, self.outch)))
        h3 = h2+pad_x

        return h3

class SingleReluBlock_addNoise(chainer.Chain):
    def __init__(self, initializer, inch, outch, util):
        self.inch = inch
        self.outch = outch
        super(SingleReluBlock_addNoise, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(inch,outch, ksize=3, stride=1, pad=1, initialW=initializer)
            util.add(self.conv1)
            self.conv2 = L.Convolution2D(outch,outch, ksize=3, stride=1, pad=1, initialW=initializer)
            util.add(self.conv2)
            self.bnorm1 = L.BatchNormalization(inch)
            self.bnorm2 = L.BatchNormalization(outch)
            self.bnorm3 = L.BatchNormalization(outch)

    def __call__(self, x, test=False):
        h0 = self.bnorm1(x, finetune=test)
        h1 = F.relu(self.bnorm2(self.conv1(h0), finetune=test))
        h2 = self.bnorm3(self.conv2(h1), finetune=test)
        h2 = G.gaussian_noise(h2,0.5)
        pad_x = F.concat((x, U.zero_pad(x, self.inch, self.outch)))
        h3 = h2+pad_x

        return h3

class SingleReluBlock_addDO(chainer.Chain):
    def __init__(self, initializer, inch, outch, util):
        self.inch = inch
        self.outch = outch
        super(SingleReluBlock_addDO, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(inch,outch, ksize=3, stride=1, pad=1, initialW=initializer)
            util.add(self.conv1)
            self.conv2 = L.Convolution2D(outch,outch, ksize=3, stride=1, pad=1, initialW=initializer)
            util.add(self.conv2)
            self.bnorm1 = L.BatchNormalization(inch)
            self.bnorm2 = L.BatchNormalization(outch)
            self.bnorm3 = L.BatchNormalization(outch)

    def __call__(self, x, test=False):
        h0 = self.bnorm1(x, finetune=test)
        h1 = self.bnorm2(self.conv1(h0), finetune=test)
        h1 = F.relu(F.dropout(h1,0.2))
        h2 = self.bnorm3(self.conv2(h1), finetune=test)
        pad_x = F.concat((x, U.zero_pad(x, self.inch, self.outch)))
        h3 = h2+pad_x

        return h3

class SingleReluBlock_addND(chainer.Chain):
    def __init__(self, initializer, inch, outch, util):
        self.inch = inch
        self.outch = outch
        super(SingleReluBlock_addND, self).__init__()
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
        h1 = self.bnorm2(self.conv1(h0), finetune=test)
        h1 = F.relu(F.dropout(h1,0.2))
        h2 = self.bnorm3(self.conv2(h1), finetune=test)
        h2 = G.gaussian_noise(h2,0.5)
        pad_x = F.concat((x, U.zero_pad(x, self.inch, self.outch)))
        h3 = h2+pad_x

        return h3

class PreActivateBlock(chainer.Chain):
    def __init__(self, initializer, inch, outch, util):
        self.inch = inch
        self.outch = outch
        super(PreActivateBlock, self).__init__()
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
        h1 = F.relu(self.bnorm2(self.conv1(h0), finetune=test))
        h2 = self.bnorm3(self.conv2(h1), finetune=test)
        pad_x = F.concat((x, U.zero_pad(x, self.inch, self.outch)))
        h3 = h2+pad_x

        return h3

class PreActivateBlockW(chainer.Chain):
    def __init__(self, initializer, inch, outch, util):
        self.inch = inch
        self.outch = outch
        super(PreActivateBlockW, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(inch,outch, ksize=3, stride=1, pad=1, initialW=initializer)
            util.add(self.conv1)
            self.conv2 = L.Convolution2D(outch,outch, ksize=3, stride=1, pad=1, initialW=initializer)
            util.add(self.conv2)
            self.bnorm1 = L.BatchNormalization(inch)
            self.bnorm2 = L.BatchNormalization(outch)
            self.bnorm3 = L.BatchNormalization(outch)

            #adder
            self.addconv = L.Convolution2D(inch+outch, outch, ksize=1, stride=1, pad=0, nobias=True, initialW=initializer)
            util.add(self.addconv, nobias=True)

    def __call__(self, x, test=False):
        h0 = F.relu(self.bnorm1(x, finetune=test))
        h1 = F.relu(self.bnorm2(self.conv1(h0), finetune=test))
        h2 = self.bnorm3(self.conv2(h1), finetune=test)
        h3 = F.concat((x,h2), axis=1)
        h = self.addconv(h3)

        return h

class PreActivateBlock_addNoise(chainer.Chain):
    def __init__(self, initializer, inch, outch, util):
        self.inch = inch
        self.outch = outch
        super(PreActivateBlock_addNoise, self).__init__()
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
        h1 = F.relu(self.bnorm2(self.conv1(h0), finetune=test))
        h2 = self.bnorm3(self.conv2(h1), finetune=test)
        h2 = G.gaussian_noise(h2,0.8)
        pad_x = F.concat((x, U.zero_pad(x, self.inch, self.outch)))
        h3 = h2+pad_x

        return h3

class PreActivateBlock_limitedNoise(chainer.Chain):
    def __init__(self, initializer, inch, outch, util, limit):
        self.inch = inch
        self.outch = outch
        self.limit = limit
        super(PreActivateBlock_limitedNoise, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(inch,outch, ksize=3, stride=1, pad=1, initialW=initializer)
            util.add(self.conv1)
            self.conv2 = L.Convolution2D(outch,outch, ksize=3, stride=1, pad=1, initialW=initializer)
            util.add(self.conv2)
            self.bnorm1 = L.BatchNormalization(inch)
            self.bnorm2 = L.BatchNormalization(outch)
            self.bnorm3 = L.BatchNormalization(outch)

    def __call__(self, x, count, test=False):
        h0 = F.relu(self.bnorm1(x, finetune=test))
        h1 = F.relu(self.bnorm2(self.conv1(h0), finetune=test))
        h2 = self.bnorm3(self.conv2(h1), finetune=test)
        if self.limit > (count/134):
            h2 = G.gaussian_noise(h2,0.8)
        pad_x = F.concat((x, U.zero_pad(x, self.inch, self.outch)))
        h3 = h2+pad_x

        return h3

class PreActivateBlock_addDO(chainer.Chain):
    def __init__(self, initializer, inch, outch, util):
        self.inch = inch
        self.outch = outch
        super(PreActivateBlock_addDO, self).__init__()
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
        h1 = self.bnorm2(self.conv1(h0), finetune=test)
        h1 = F.relu(F.dropout(h1,0.2))
        h2 = self.bnorm3(self.conv2(h1), finetune=test)
        pad_x = F.concat((x, U.zero_pad(x, self.inch, self.outch)))
        h3 = h2+pad_x

        return h3

class WalpurgisNightBlock(chainer.Chain):
    def __init__(self, initializer, inch, outch, util):
        self.inch = inch
        self.outch = outch
        super(WalpurgisNightBlock, self).__init__()
        with self.init_scope():
            #plain
            self.pconv1 = L.Convolution2D(inch,outch, ksize=3, stride=1, pad=1, initialW=initializer)
            util.add(self.pconv1)
            self.pconv2 = L.Convolution2D(outch,outch, ksize=3, stride=1, pad=1, initialW=initializer)
            util.add(self.pconv2)
            self.pbnorm1 = L.BatchNormalization(inch)
            self.pbnorm2 = L.BatchNormalization(outch)
            self.pbnorm3 = L.BatchNormalization(outch)

            #noise
            self.nconv1 = L.Convolution2D(inch,outch, ksize=3, stride=1, pad=1, initialW=initializer)
            util.add(self.nconv1)
            self.nconv2 = L.Convolution2D(outch,outch, ksize=3, stride=1, pad=1, initialW=initializer)
            util.add(self.nconv2)
            self.nbnorm1 = L.BatchNormalization(inch)
            self.nbnorm2 = L.BatchNormalization(outch)
            self.nbnorm3 = L.BatchNormalization(outch)

            #Dropout
            self.dconv1 = L.Convolution2D(inch,outch, ksize=3, stride=1, pad=1, initialW=initializer)
            util.add(self.dconv1)
            self.dconv2 = L.Convolution2D(outch,outch, ksize=3, stride=1, pad=1, initialW=initializer)
            util.add(self.dconv2)
            self.dbnorm1 = L.BatchNormalization(inch)
            self.dbnorm2 = L.BatchNormalization(outch)
            self.dbnorm3 = L.BatchNormalization(outch)

            #adder
            self.addconv = L.Convolution2D(inch+(outch*3), outch, ksize=1, stride=1, pad=0, nobias=True, initialW=initializer)
            util.add(self.addconv, nobias=True)

    def __call__(self, x, test=False):
        #plain
        p0 = F.relu(self.pbnorm1(x, finetune=test))
        p1 = F.relu(self.pbnorm2(self.pconv1(p0), finetune=test))
        p2 = self.pbnorm3(self.pconv2(p1), finetune=test)

        #noise
        n0 = F.relu(self.nbnorm1(x, finetune=test))
        n1 = F.relu(self.nbnorm2(self.nconv1(n0), finetune=test))
        n2 = self.nbnorm3(self.nconv2(n1), finetune=test)
        n2 = G.gaussian_noise(n2,0.5)

        #dropout
        d0 = F.relu(self.dbnorm1(x, finetune=test))
        d1 = self.dbnorm2(self.dconv1(d0), finetune=test)
        d1 = F.relu(F.dropout(d1,0.2))
        d2 = self.dbnorm3(self.dconv2(d1), finetune=test)

        h = F.concat((x, p2, n2, d2), axis=1)

        return self.addconv(h)

class SingleReluBlock_addNoiseW(chainer.Chain):
    def __init__(self, initializer, inch, outch, util):
        self.inch = inch
        self.outch = outch
        super(SingleReluBlock_addNoiseW, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(inch,outch, ksize=3, stride=1, pad=1, initialW=initializer)
            util.add(self.conv1)
            self.conv2 = L.Convolution2D(outch,outch, ksize=3, stride=1, pad=1, initialW=initializer)
            util.add(self.conv2)
            self.bnorm1 = L.BatchNormalization(inch)
            self.bnorm2 = L.BatchNormalization(outch)
            self.bnorm3 = L.BatchNormalization(outch)

            #adder
            self.addconv = L.Convolution2D(inch+outch, outch, ksize=1, stride=1, pad=0, nobias=True, initialW=initializer)
            util.add(self.addconv, nobias=True)

    def __call__(self, x, test=False):
        h0 = self.bnorm1(x, finetune=test)
        h1 = F.relu(self.bnorm2(self.conv1(h0), finetune=test))
        h2 = self.bnorm3(self.conv2(h1), finetune=test)
        h2 = G.gaussian_noise(h2,0.5)
        h3 = F.concat((x,h2), axis=1)
        h = self.addconv(h3)

        return h
