# coding: utf-8
#chainer
import os, sys
from . import utils
from . import learn
from . import export
from . import dataset
import argparse
import chainer.initializers as I
from chainer import cuda
import numpy as np

class Trainer(object):
    def __init__(cls, filename, export_dir, dataset_dir):
        cls.filename = filename
        cls.export_path = os.path.join(export_dir,filename)

        cls.args = cls.set_args()
        cls.dataset = dataset.feeder(dataset_dir, cls.gpu_use)
        cls.init_print()

    def set_args(cls):
        #コマンドライン引数の解析と読み込み
        parser = argparse.ArgumentParser(description='Chainer traning: IRMAS')
        parser.add_argument('epochs', metavar='N', type=int, nargs='+',help='List of testing epochs')
        parser.add_argument('--maximum', '-m', type=int, default=-1, help='Save the models that leave maximum F1 values')
        parser.add_argument('--gpu', '-g', type=int, default=-1, help='GPU ID (negative value indicates CPU)')
        parser.add_argument('--ensemble', '-n', type=int, default=1, help='value of use model')
        parser.add_argument('--learning_rate', '-l', type=float, default=0.01, help='value of learning rate')
        parser.add_argument('--weight_decay', '-w', type=float, default=0.0001, help='value of weight decay')
        parser.add_argument('--importing', '-i', type=int, default=0, help='flag of importing trained models[true/1 false/0]')
        args = parser.parse_args()

        cls.gpu_use = args.gpu >= 0
        cls.import_model = args.importing > 0
        return args

    def init_print(cls):
        guse = "True" if cls.gpu_use else "False"
        imp = "True" if cls.import_model else "False"
        print("["+cls.filename+"]", end="\n\n")
        print("# GPU use : "+guse)
        print("# model quantity : %d" % cls.args.ensemble)
        print("# training epochs : ",cls.args.epochs)
        print("# learning rate : %f" % cls.args.learning_rate)
        print("# weight decay value : %f" % cls.args.weight_decay)
        print("# import trained model : "+imp)

        if cls.args.maximum >= 0:
            print("# save max F1 values since %d epoch" % cls.args.maximum)

        print("")

    def set_networks(cls, classifier, initializer=None):
        if initializer is None:
            initializer = I.HeNormal
        cls.initializer = initializer
        cls.classifier = classifier

    def model_init(cls):
        ensemble = cls.args.ensemble
        epochs = cls.args.epochs
        lr = cls.args.learning_rate
        wd = cls.args.weight_decay
        import_model = cls.import_model
        train_epochs, train_logs, model, optimizer = utils.train_init(ins=cls, ensemble=ensemble, epochs=epochs, lr=lr, wd=wd, import_model=import_model)
        cls.model = model
        cls.optimizer = optimizer
        cls.epochs = train_epochs
        cls.logs = train_logs

    def train_loop(cls, lr_update=None):
        if lr_update is not None:
            nlr_list = np.zeros(cls.args.epochs)
            for i in range(len(lr_update[0])):
                nlr_list[lr_update[0][i]:] = lr_update[1][i]
            lr_list = nlr_list.tolist()
        else:
            lr_list = None

        for i in range(cls.args.ensemble):
            log = []
            if len(cls.logs) > 0:
                log = cls.logs[i]

            utils.loop(cls.dataset, cls.epochs[i], log, cls.args.maximum, cls.args.gpu, cls.model[i], cls.optimizer[i], i, cls.export_path, lr_list)

    def testing(cls):
        print("final testing ....\r",end="")
        test_x, labels = learn.ensemble(2, cls.dataset, 6, cls.model)
        test_result ,result_list = learn.validation(test_x, labels, only_max=False)
        cls.result = test_result
        export.evaluation_plot(result_list, "result", cls.export_path)

        print("final test is done!")
        if cls.args.maximum >= 0:
            print("max model testing ....\r",end="")
            max_model = cls.model
            for i in range(cls.args.ensemble):
                #maxモデルのインポート
                model_name = ("max_model%02d.npz" % i)
                model_path = os.path.join(cls.export_path, model_name)
                if os.path.exists(model_path):
                    max_model[i] = export.model_import(model_path, max_model[i], cls.args.gpu)

            max_test_x, labels = learn.ensemble(2, cls.dataset, 6, cls.model)
            max_test_result ,max_result_list = learn.validation(max_test_x, labels, only_max=False)
            cls.result.extend(max_test_result)
            export.evaluation_plot(max_result_list, "max_result", cls.export_path)

            print("max model test is done!")


    def export(cls):
        train = []
        epoch = []
        for i in range(cls.args.ensemble):
            dict_log = export.pkl_import(cls.export_path, i)
            train.append(dict_log["log"])
            epoch.append(dict_log["epoch"])

        export.log_export(cls.args.ensemble, train, cls.result, cls.model[0], cls.filename, cls.export_path, max(epoch))
