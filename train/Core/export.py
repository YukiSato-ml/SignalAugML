# coding: utf-8
#chainer

import os
import sys
import pickle
import chainer
from chainer import serializers
import numpy as np

def snapshot_export(epoch, m_num, model, log, gpu, snapshot_dir):

    if not os.path.isdir(snapshot_dir):
        os.mkdir(snapshot_dir)

    print("outputting the snapshots......\r", end="")
    #モデルの出力
    model_name = ("model%02d.npz" % m_num)
    model_path = os.path.join(snapshot_dir, model_name)
    model = model_export(model_path, model, gpu)

    #ログ（pickleで圧縮）の出力
    dict_log = {"epoch":epoch, "log":log}
    pkl_export(snapshot_dir, m_num, dict_log)

    #グラフの出力
    fig_plot(log, m_num, snapshot_dir)
    print("output the snapshots - No.%02d" % m_num)

def snapshot_import(m_num, model, gpu, snapshot_dir):
    print("importing the snapshots......\r", end="")
    model_name = ("model%02d.npz" % m_num)
    model_path = os.path.join(snapshot_dir, model_name)
    if os.path.exists(model_path):
        model = model_import(model_path, model, gpu)

    dict_log = pkl_import(snapshot_dir, m_num)
    print("import the snapshots - No.%02d" % m_num)
    return model, dict_log

def pkl_export(snapshot_dir, m_num, dict_log):
    log_name = ("logs%02d.pkl" % m_num)
    log_path = os.path.join(snapshot_dir, log_name)
    with open(log_path, "wb") as f:
        pickle.dump(dict_log, f)

def pkl_import(snapshot_dir, m_num):
    log_name = ("logs%02d.pkl" % m_num)
    log_path = os.path.join(snapshot_dir, log_name)
    if os.path.exists(log_path):
        with open(log_path, "rb") as f:
            dict_log = pickle.load(f)
    else:
        dict_log = {"epoch":-1, "log":[]}
    return dict_log

def model_export(model_path, model, gpu):
    if gpu >= 0:
        model.to_cpu() # CPUで計算できるようにしておく
    serializers.save_npz(model_path, model) # npz形式で書き出し
    if gpu >= 0:
        chainer.cuda.get_device_from_id(gpu).use()
        model.to_gpu()  # GPUで計算できるように治す
    return model

def model_import(model_path, model, gpu):
    serializers.load_npz(model_path, model)
    if gpu >= 0:
        chainer.cuda.get_device_from_id(gpu).use()
        model.to_gpu()  # GPUで計算できるように治す
    return model

def log_export(m_num, train, test, model, filename, log_path, epoch_num):
    print("global log exporting....\r", end="")
    import csv
    ensemble = [""]
    for i in range(m_num):
        ensemble.extend(["model%d"%i,"","","","","",""])
        fig_plot(train[i], i, log_path)

    log_name = (filename+"_log.csv")
    log_path = os.path.join(log_path,log_name)
    with open(log_path, 'w') as f:
        writer = csv.writer(f, lineterminator='\n') # 改行コード（\n）を指定しておく
        writer.writerow([filename])
        writer.writerow(ensemble)
        count = 0
        for e in range(epoch_num+2):
            if e == 0:
                epochs = ["epoch"]
            else:
                epochs = [e-1]
            for i in range(m_num):
                if e >= len(train[i]) :
                    epochs.extend(["--","--","--","--","--","--"])
                else:
                    epochs.extend(train[i][e])
                epochs.extend([""])
            writer.writerow(epochs)     # list（1次元配列）の場合

        #最高値の保存,平均の計算
        train_result = []
        for i in range(m_num):
            train_result_tmp = []
            train_result_tmp.append(["[max]","","","","","",""])
            train_result_tmp.append(["epoch","threshold","P value","R value","F value","",""])
            ntrain = np.array(train[i][1:])
            max_n = ntrain[:, 5].max()#F値の最高値
            max_i = ntrain[:, 5].tolist().index(max_n)+1#最高値のインデックス番地

            max_tmp = [max_i-1]
            max_tmp.extend(train[i][max_i][2:])
            max_tmp.extend(["",""])
            train_result_tmp.append(max_tmp)
            train_result_tmp.append(["[average]","epoch","P value","R value","F value","",""])

            ave_len = int(len(train[i][1:])/2)
            ave_list = np.array(train[i][ave_len:])[:, 3:].mean(axis=0).tolist()
            ave_tmp = [""]
            ave_tmp.extend([ave_len])
            ave_tmp.extend(ave_list)
            ave_tmp.extend(["",""])
            train_result_tmp.append(ave_tmp)

            train_result.append(train_result_tmp)

        for i in range(5):
            tmp_list = [""]
            for e in range(m_num):
                tmp_list.extend(train_result[e][i])
            writer.writerow(tmp_list)

        #test結果の記入
        writer.writerow("")
        writer.writerow(["testing result"])
        writer.writerow(["threshold", "P value", "R value", "F value"])
        writer.writerow(test[:4])
        if len(test) > 4:
            writer.writerow("")
            writer.writerow(["max model result"])
            writer.writerow(["threshold", "P value", "R value", "F value"])
            writer.writerow(test[4:])

        #パラメータ数の記入
        param_num, param_sum, param_list = model.param_count()
        writer.writerow(" ")
        writer.writerow(["layer num",param_num])
        writer.writerow(["parameter num", param_sum])
        writer.writerow(["parameter num list"])
        writer.writerow(param_list)
    print("global log is already exported!")

def fig_plot(log, m_num, export_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n_log = np.array(log[1:])
    loss = n_log[:,0]
    accuracy = n_log[:,1]
    p_value = n_log[::5,3]
    r_value = n_log[::5,4]
    f_value = n_log[::5,5]

    eva_x = np.arange(0,len(loss), 5)

    plt.figure()
    plt.title("model%02d result" % m_num, loc='center')
    plt.plot(loss, 'b', label='loss')
    plt.plot(accuracy, 'r', label='accuracy')
    plt.plot(eva_x,p_value, 'y', label='P value')
    plt.plot(eva_x,r_value, 'c', label='R value')
    plt.plot(eva_x,f_value, 'g:', label='F value')
    plt.xlim([0, len(log)])
    plt.xticks(np.arange(0, len(log), 50, dtype='int'))
    plt.ylim([0, 1.05])
    plt.yticks(np.arange(0.0, 1.05, 0.05))
    plt.grid(color='gray')
    plt.legend(loc=3)


    fig_name = ("model%02d_fig.png" % m_num)
    fig_path = os.path.join(export_dir, fig_name)
    plt.savefig(fig_path)

def evaluation_plot(result, name, export_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nresu = np.array(result)
    values_th = nresu[:,0]
    values_P = nresu[:,1]
    values_R = nresu[:,2]
    values_F = nresu[:,3]

    plt.figure()
    plt.title("%s" % name, loc='center')
    plt.plot(values_th, values_P, 'b', label='P value')
    plt.plot(values_th,values_R, 'g', label='R value')
    plt.plot(values_th,values_F, 'r--', label='F value')
    plt.xticks(np.arange(0.0, 1.0, 0.1))
    plt.ylim([0, 1.05])
    plt.yticks(np.arange(0.0, 1.0, 0.1))
    plt.grid(color='gray')
    plt.legend(loc=3)

    fig_name = ("%s_fig.png" % name)
    fig_path = os.path.join(export_dir, fig_name)
    plt.savefig(fig_path)
