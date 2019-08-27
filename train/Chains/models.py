# coding: utf-8
import chainer

from . import blocks
from . import chain
from . import classifier as C

'''
res  def
layer 10
32-32x2-64x2-128x2-256x2-11
conv 1 + 2 x 4
fully 1
max pooling
ReLU
'''
class ResidualNN01(C.Classifier):
    layer_num = [32,32,64,128,256]
    def __init__(self,initializer, layer_num=layer_num):
        super().__init__(chain.Res4Chain(initializer,blocks.ResBlock, layer_num))

'''
res gauss nois
layer 10
32-32x2-64x2-128x2-256x2-11
conv 1 + 2 x 4
fully 1
max pooling
ReLU
'''
class ResidualNN02(C.Classifier):
    layer_num = [32,32,64,128,256]
    def __init__(self,initializer, layer_num=layer_num):
        super().__init__(chain.Res4Chain(initializer,blocks.ResBlock_addNoise, layer_num))

'''
res drop
layer 10
32-32x2-64x2-128x2-256x2-11
conv 1 + 2 x 4
fully 1
max pooling
dropout 20%
ReLU
'''
class ResidualNN03(C.Classifier):
    layer_num = [32,32,64,128,256]
    def __init__(self,initializer, layer_num=layer_num):
        super().__init__(chain.Res4Chain(initializer,blocks.ResBlock_addDO, layer_num))

'''
Single ReLU
layer 10
32-32x2-64x2-128x2-256x2-11
conv 1 + 2 x 4
fully 1
max pooling
ReLU
'''
class ResidualNN04(C.Classifier):
    layer_num = [32,32,64,128,256]
    def __init__(self,initializer, layer_num=layer_num):
        super().__init__(chain.Res4Chain(initializer,blocks.SingleReluBlock, layer_num, initialBN=True))

'''
res  def
layer 12
32-32x2-64x2-128x2-256x2-512x2-11
conv 1 + 2 x 5
fully 1
max pooling
ReLU
'''
class ResidualNN05(C.Classifier):
    layer_num = [32,32,64,128,256,512]
    def __init__(self,initializer, layer_num=layer_num):
        super().__init__(chain.Res5Chain(initializer,blocks.ResBlock, layer_num))

'''
res gauss nois
layer 12
32-32x2-64x2-128x2-256x2-512x2-11
conv 1 + 2 x 5
fully 1
max pooling
ReLU
'''
class ResidualNN06(C.Classifier):
    layer_num = [32,32,64,128,256,512]
    def __init__(self,initializer, layer_num=layer_num):
        super().__init__(chain.Res5Chain(initializer,blocks.ResBlock_addNoise, layer_num))

'''
res drop
layer 12
32-32x2-64x2-128x2-256x2-512x2-11
conv 1 + 2 x 5
fully 1
max pooling
dropout 20%
ReLU
'''
class ResidualNN07(C.Classifier):
    layer_num = [32,32,64,128,256,512]
    def __init__(self,initializer, layer_num=layer_num):
        super().__init__(chain.Res5Chain(initializer,blocks.ResBlock_addDO, layer_num))

'''
Single ReLU
layer 12
32-32x2-64x2-128x2-256x2-512x2-11
conv 1 + 2 x 5
fully 1
max pooling
ReLU
'''
class ResidualNN08(C.Classifier):
    layer_num = [32,32,64,128,256,512]
    def __init__(self,initializer, layer_num=layer_num):
        super().__init__(chain.Res5Chain(initializer,blocks.SingleReluBlock, layer_num, initialBN=True))

'''
res  def
layer 17
32-32x2x2-64x2x2-128x2x2-256x2x2-11
conv 1 + 2 x 8
fully 1
max pooling
ReLU
'''
class ResidualNN09(C.Classifier):
    layer_num = [32,32,64,128,256]
    def __init__(self,initializer, layer_num=layer_num):
        super().__init__(chain.Res8Chain(initializer,blocks.ResBlock, layer_num))

'''
res gauss nois
layer 17
32-32x2x2-64x2x2-128x2x2-256x2x2-11
conv 1 + 2 x 8
fully 1
max pooling
ReLU
'''
class ResidualNN10(C.Classifier):
    layer_num = [32,32,64,128,256]
    def __init__(self,initializer, layer_num=layer_num):
        super().__init__(chain.Res8Chain(initializer,blocks.ResBlock_addNoise, layer_num))

'''
res drop
layer 17
32-32x2x2-64x2x2-128x2x2-256x2x2-11
conv 1 + 2 x 8
fully 1
max pooling
dropout 20%
ReLU
'''
class ResidualNN11(C.Classifier):
    layer_num = [32,32,64,128,256]
    def __init__(self,initializer, layer_num=layer_num):
        super().__init__(chain.Res8Chain(initializer,blocks.ResBlock_addDO, layer_num))

'''
Single ReLU
layer 17
32-32x2x2-64x2x2-128x2x2-256x2x2-11
conv 1 + 2 x 8
fully 1
max pooling
ReLU
'''
class ResidualNN12(C.Classifier):
    layer_num = [32,32,64,128,256]
    def __init__(self,initializer, layer_num=layer_num):
        super().__init__(chain.Res8Chain(initializer,blocks.SingleReluBlock, layer_num, initialBN=True))

'''
res  def
layer 21
32-32x2x2-64x2x2-128x2x2-256x2x2-512x2x2-11
conv 1 + 2 x 10
fully 1
max pooling
ReLU
'''
class ResidualNN13(C.Classifier):
    layer_num = [32,32,64,128,256,512]
    def __init__(self,initializer, layer_num=layer_num):
        super().__init__(chain.Res10Chain(initializer,blocks.ResBlock, layer_num))

'''
res gauss nois
layer 21
32-32x2x2-64x2x2-128x2x2-256x2x2-512x2x2-11
conv 1 + 2 x 10
fully 1
max pooling
ReLU
'''
class ResidualNN14(C.Classifier):
    layer_num = [32,32,64,128,256,512]
    def __init__(self,initializer, layer_num=layer_num):
        super().__init__(chain.Res10Chain(initializer,blocks.ResBlock_addNoise, layer_num))

'''
res drop
layer 21
32-32x2x2-64x2x2-128x2x2-256x2x2-512x2x2-11
conv 1 + 2 x 10
fully 1
max pooling
dropout 20%
ReLU
'''
class ResidualNN15(C.Classifier):
    layer_num = [32,32,64,128,256,512]
    def __init__(self,initializer, layer_num=layer_num):
        super().__init__(chain.Res10Chain(initializer,blocks.ResBlock_addDO, layer_num))

'''
Single ReLU
layer 21
32-32x2x2-64x2x2-128x2x2-256x2x2-512x2x2-11
conv 1 + 2 x 10
fully 1
max pooling
ReLU
'''
class ResidualNN16(C.Classifier):
    layer_num = [32,32,64,128,256,512]
    def __init__(self,initializer, layer_num=layer_num):
        super().__init__(chain.Res10Chain(initializer,blocks.SingleReluBlock, layer_num, initialBN=True))

'''
res  def
layer 33
32-32x2x4-64x2x4-128x2x4-256x2x4-11
conv 1 + 2 x 16
fully 1
max pooling
ReLU
'''
class ResidualNN17(C.Classifier):
    layer_num = [32,32,64,128,256]
    def __init__(self,initializer, layer_num=layer_num):
        super().__init__(chain.Res16Chain(initializer,blocks.ResBlock, layer_num))

'''
res gauss nois
layer 33
32-32x2x4-64x2x4-128x2x4-256x2x4-11
conv 1 + 2 x 16
fully 1
max pooling
ReLU
'''
class ResidualNN18(C.Classifier):
    layer_num = [32,32,64,128,256]
    def __init__(self,initializer, layer_num=layer_num):
        super().__init__(chain.Res16Chain(initializer,blocks.ResBlock_addNoise, layer_num))

'''
res drop
layer 33
32-32x2x4-64x2x4-128x2x4-256x2x4-11
conv 1 + 2 x 16
fully 1
max pooling
dropout 20%
ReLU
'''
class ResidualNN19(C.Classifier):
    layer_num = [32,32,64,128,256]
    def __init__(self,initializer, layer_num=layer_num):
        super().__init__(chain.Res16Chain(initializer,blocks.ResBlock_addDO, layer_num))

'''
Single ReLU
layer 33
32-32x2x4-64x2x4-128x2x4-256x2x4-11
conv 1 + 2 x 16
fully 1
max pooling
ReLU
'''
class ResidualNN20(C.Classifier):
    layer_num = [32,32,64,128,256]
    def __init__(self,initializer, layer_num=layer_num):
        super().__init__(chain.Res16Chain(initializer,blocks.SingleReluBlock, layer_num, initialBN=True))

'''
res  def
layer 42
32-32x2x4-64x2x4-128x2x4-256x2x4-512x2x4-11
conv 1 + 2 x 20
fully 1
max pooling
ReLU
'''
class ResidualNN21(C.Classifier):
    layer_num = [32,32,64,128,256,512]
    def __init__(self,initializer, layer_num=layer_num):
        super().__init__(chain.Res20Chain(initializer,blocks.ResBlock, layer_num))

'''
res gauss nois
layer 42
32-32x2x4-64x2x4-128x2x4-256x2x4-512x2x4-11
conv 1 + 2 x 20
fully 1
max pooling
ReLU
'''
class ResidualNN22(C.Classifier):
    layer_num = [32,32,64,128,256,512]
    def __init__(self,initializer, layer_num=layer_num):
        super().__init__(chain.Res20Chain(initializer,blocks.ResBlock_addNoise, layer_num))

'''
res drop
layer 42
32-32x2x4-64x2x4-128x2x4-256x2x4-512x2x4-11
conv 1 + 2 x 20
fully 1
max pooling
dropout 20%
ReLU
'''
class ResidualNN23(C.Classifier):
    layer_num = [32,32,64,128,256,512]
    def __init__(self,initializer, layer_num=layer_num):
        super().__init__(chain.Res20Chain(initializer,blocks.ResBlock_addDO, layer_num))

'''
Single ReLU
layer 42
32-32x2x4-64x2x4-128x2x4-256x2x4-512x2x4-11
conv 1 + 2 x 20
fully 1
max pooling
ReLU
'''
class ResidualNN24(C.Classifier):
    layer_num = [32,32,64,128,256,512]
    def __init__(self,initializer, layer_num=layer_num):
        super().__init__(chain.Res20Chain(initializer,blocks.SingleReluBlock, layer_num, initialBN=True))

'''
Single ReLU noise
layer 10
32-32x2-64x2-128x2-256x2-11
conv 1 + 2 x 4
fully 1
max pooling
ReLU
'''
class ResidualNN25(C.Classifier):
    layer_num = [32,32,64,128,256]
    def __init__(self,initializer, layer_num=layer_num):
        super().__init__(chain.Res4Chain(initializer,blocks.SingleReluBlock_addNoise, layer_num, initialBN=True))

'''
Single ReLU dropout
layer 10
32-32x2-64x2-128x2-256x2-11
conv 1 + 2 x 4
fully 1
max pooling
ReLU
'''
class ResidualNN26(C.Classifier):
    layer_num = [32,32,64,128,256]
    def __init__(self,initializer, layer_num=layer_num):
        super().__init__(chain.Res4Chain(initializer,blocks.SingleReluBlock_addDO, layer_num, initialBN=True))
