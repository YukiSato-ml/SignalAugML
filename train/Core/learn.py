# coding: utf-8
#chainer

import sys
import chainer
from chainer import cuda
import numpy as np

def train(feed, model, optimizer):
    chainer.config.train = True
    loss_log = []
    accuracy_log = []
    loss_sum     = 0
    accuracy_sum = 0
    count = 0
    for d in feed:

        model.zerograds()
        loss, accuracy = model.forword(d['data'], d['label'], d['size'])
        loss.backward()
        optimizer.update()

        print("%03d loss: %1.5f accuracy: %1.5f \r" % (count, loss.data, accuracy.data), end="")
        sys.stdout.flush()
        loss_log.append(float(loss.data))
        accuracy_log.append(float(accuracy.data))
        loss_sum     += loss.data
        accuracy_sum += accuracy.data
        count += 1

    print("ehoch result loss:",loss_sum/count ,"  accuracy:",accuracy_sum/count)
    return [float(loss_sum/count), float(accuracy_sum/count)]

#単一モデルのためのtest
def test(feed, maxpool, model):
    chainer.config.train = False

    count = 0
    for d in feed:
        prediction = model.testing(d['data'], d['size'], maxpool)
        if 'x_prodict' not in locals():
            xp  = cuda.get_array_module(*prediction)
            x_prodict = xp.empty(0)
            y_labels  = xp.empty(0)
        x_prodict = xp.concatenate((x_prodict, prediction))
        y_labels  = xp.concatenate((y_labels , d['label']))
        count += 1
    return x_prodict.reshape((count,-1)) ,y_labels.reshape((count,-1))

#複数モデルのためのtest
def ensemble(mode, data, maxpool, model):
    chainer.config.train = False
    experiment = len(model)

    for j in range(experiment):
        feed = data.test_feed(mode)
        count = 0
        for d in feed:
            prediction = model[j].testing(d['data'], d['size'], maxpool)
            if 'x_prodict' not in locals():
                xp  = cuda.get_array_module(*prediction)
                x_prodict = xp.empty(0)
                y_labels  = xp.empty(0)
            x_prodict = xp.concatenate((x_prodict,prediction))
            y_labels  = xp.concatenate((y_labels , d['label']))
            count += 1

    x_prodict = x_prodict.reshape((experiment, -1))
    y_labels  = y_labels.reshape((experiment, -1))
    return x_prodict.mean(axis=0).reshape((count,-1)), y_labels[0].reshape((count,-1))


def F1measure(y, t):
    tp = (y & t).astype(int)
    fp = (y & (~t)).astype(int)
    fn = ((~y) & t).astype(int)
    tn = (~(y & t)).astype(int)
    test_num = len(y)
    label_num = len(y[0])
    correct = 0.00001

    #ミクロ値
    micro_tpc = tp.sum(axis=1)
    micro_fpc = fp.sum(axis=1)
    micro_fnc = fn.sum(axis=1)

    p_micro = (micro_tpc/(micro_tpc+micro_fpc+correct)).sum()/test_num
    r_micro = (micro_tpc/(micro_tpc+micro_fnc+correct)).sum()/test_num
    F1_micro = (2*p_micro*r_micro)/(p_micro+r_micro)

    micro = {}
    micro['P'] = p_micro
    micro['R'] = r_micro
    micro['F1'] = F1_micro

    #マクロ値
    macro_tpc = tp.sum(axis=0)
    macro_fpc = fp.sum(axis=0)
    macro_fnc = fn.sum(axis=0)

    p_macro = (macro_tpc/(macro_tpc+macro_fpc+correct)).sum()/label_num
    r_macro = (macro_tpc/(macro_tpc+macro_fnc+correct)).sum()/label_num
    F1_macro = (2*p_macro*r_macro)/(p_macro+r_macro)

    macro = {}
    macro['P'] = p_macro
    macro['R'] = r_macro
    macro['F1'] = F1_macro

    return micro, macro

def evaluation(x, y, threshold):
    xp  = cuda.get_array_module(*x)
    test_num = len(x)

    mic_P = 0
    mic_R = 0
    mic_F = 0
    mac_P = 0
    mac_R = 0
    mac_F = 0
    #bool化 yは0or1なので　閾値を0.5にする
    t = y > 0.5

    t = xp.array(t)

    #bool化
    eva_x = x >= threshold
    eva_x = eva_x.astype(xp.int32)
    micro,macro = F1measure(eva_x, t)
    mic_P += micro['P']
    mic_R += micro['R']
    mic_F += micro['F1']
    mac_P += macro['P']
    mac_R += macro['R']
    mac_F += macro['F1']

    return [0 ,float(mic_P), float(mic_R), float(mic_F), mac_P, mac_R, mac_F]

def validation(test_x, labels, only_max = True):
    threshold = np.arange(0.05,0.95,0.01)
    threshold = threshold.tolist()
    min_F_max = 0
    result=[]
    for th in threshold:
        tmp_result = evaluation(test_x, labels, th)
        tmp_result[0] = th
        result.append(tmp_result)
    nF = np.array(result)[:, 3]
    max_f = nF.max()
    max_i = nF.tolist().index(max_f)

    if only_max:
        return result[max_i][:4]
    else:
        return result[max_i][:4] ,result
