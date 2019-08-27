# coding: utf-8
#chainer

import sys
import os
import chainer
from chainer import cuda
import numpy as np
from . import learn
from . import export

def train_init(ins, ensemble, epochs, lr=0.01, wd=0.0001, import_model=False):
    #ネットワーク構成を定義
    model = []
    optimizer = []

    for i in range(ensemble):
        model.append(ins.classifier(ins.initializer()))
        if ins.gpu_use:
            # Make a specified GPU current
            chainer.cuda.get_device_from_id(ins.args.gpu).use()
            model[i].to_gpu()  # Copy the model to the GPU

        #optimizerのセット
        optimizer.append(chainer.optimizers.MomentumSGD(lr=lr))
        optimizer[i].setup(model[i])
        optimizer[i].add_hook(chainer.optimizer.WeightDecay(wd))

    epoch_lists = np.ones(ensemble)
    epoch_lists = np.int_(epoch_lists * np.array(epochs))
    epoch_lists = epoch_lists.tolist()

    #学習済みモデルのロード  及びエポックリストの作成
    train_logs = []
    train_epochs = []
    if import_model:
        for i in range(ensemble):
            model[i], dict_log = export.snapshot_import(i, model[i], ins.args.gpu, ins.export_path)
            train_logs.append(dict_log['log'])
            if dict_log["epoch"]+1 >= epoch_lists[i]:
                tmp_list = []
            else:
                tmp_list = np.arange(dict_log["epoch"]+1,epoch_lists[i], dtype="int").tolist()
            train_epochs.append(tmp_list)
    else:
        for i in range(ensemble):
            tmp_list = np.arange(0,epoch_lists[i], dtype="int").tolist()
            train_epochs.append(tmp_list)
            train_logs.append([])

    return train_epochs, train_logs, model, optimizer


def loop(data, epoch, log, maximum, gpu, model, optimizer, m_num, snapshot_dir, lr_list):
    test_result = [0,0,0,0]

    if len(log) is 0:
        log=[]
        log.append(["loss","accuracy", "threshold", "P value", "R value", "F1 value"])

    if len(log) > 1:
        test_result = log[-1][2:]

    print("training start -- model%02d" % m_num)
    print("Learning Rate : %f" % optimizer.hyperparam.lr)
    for e in epoch:
        #トレーニング
        print("epoch : %d" % e)
        if lr_list is not None:#learog rateの更新
            if optimizer.hyperparam.lr - lr_list[e] != 0:
                print("update model%d's Learning Rate : %f" %(m_num, optimizer.hyperparam.lr))
            optimizer.hyperparam.lr = lr_list[e]
        feed = data.train_feed(batchsize = 128)#batchsizeは可変
        result = learn.train(feed, model, optimizer)

        #テスト
        if e%5 == 0:
            print("validation testing......\r", end="")
            feed = data.test_feed(0)
            test_x, labels = learn.test(feed, 6, model)
            test_result = learn.validation(test_x, labels)
            print("validation test is done")
            print("[test] threshold:%1.2f P:%1.4f R:%1.4f F:%1.4f" %(test_result[0], test_result[1], test_result[2], test_result[3]))
            print(" ")
            if maximum <= e:
                prev_max = np.array(log[1:])[:, 5].max()
                if prev_max < test_result[3]:
                    print("max result model exporting......\r", end="")
                    model_name = ("max_model%02d.npz" % m_num)
                    model_path = os.path.join(snapshot_dir, model_name)
                    export.model_export(model_path, model, gpu)
                    print("max result model export")

        #ログを残す
        result.extend(test_result)
        log.append(result)
        if e%20 == 0:
            export.snapshot_export(e, m_num, model, log, gpu, snapshot_dir)

        #エポックログのファイル出力
        log_path = os.path.join(snapshot_dir+"epoch_log.txt")
        with open(log_path, 'w') as f:
            print("model num:%d  epoch num:%04d" %(m_num, e), file=f)
            print("  loss:%1.4f accuracy:%1.4f" %(result[0], result[1]), file=f)
            print("  threshold:%1.2f P:%1.4f R:%1.4f F:%1.4f" %(test_result[0], test_result[1], test_result[2], test_result[3]), file=f)
            print("------------", file=f)
    if len(epoch) > 0:
        export.snapshot_export(epoch[-1], m_num, model, log, gpu, snapshot_dir)
