# coding: utf-8
#chainer
import os
import wave
import glob
import numpy as np
import pickle
from scipy import fromstring, int16
from scipy.fftpack import fft
from chainer import cuda

from . import process

#ファイルパス
write_dir = "pickles"

dataset_dir = {
        'train':'data/IRMAS-TrainingData',
        'test':'data/IRMAS-TestingData'
}

label_name = ['cel', 'cla', 'flu', 'gac', 'gel', 'org', 'pia', 'sax', 'tru', 'vio', 'voi']
test_dir = ['Part1', 'Part2', 'Part3']

train_dump = 'train_set.pkl'
test_dump = 'test_set_{0}.pkl'
setup = "setparam.pkl"


sampling = 22050
datasize = 43*128

class Convert :
    def __init__(self, path):
        invide = 1
        down_sampling = True
        wave_pram = [invide, down_sampling]

        win = 1024
        ft_pram = [win]

        ch = 128
        mfb_pram = [ch]

        self.dirpath = path
        self.label_size = 11
        self.proc = process.AudioProcess(wave_pram, ft_pram)#, mfb_pram)

    def __call__(self):
        print("train data Converting ...")
        self.train_datas()
        print("Done                            \n")

        print("test data Converting ...")
        self.test_datas()
        print("Done                            \n")

        print("system param dumping ...")
        self.sys_dump()


    def train_datas(self):
        path = os.path.join(self.dirpath, dataset_dir['train'])
        #ラベル・データを保存するリストを用意
        label_save = []
        data_save = []

        #ラベルループ
        count = 0
        for label in label_name:
            labelpath = os.path.join(path, label)
            filename = os.listdir(labelpath)

            vector = np.zeros(self.label_size)
            vector[count] = 1

            #ファイルループ
            file_count = 0
            for filepath in filename:
                print(" train file (%s) %5d/%5d \r" %(label, file_count, len(filename)), end="")
                file_count += 1
                self.proc(os.path.join(labelpath,filepath))
                for data in self.proc.processing():
                    #リストに変換して保持
                    label_save.append(vector.tolist())
                    data_save.append(data.tolist())

            count += 1

        self.dumping(train_dump, data_save, label_save)

    def test_datas(self):
        #ディレクトリループ
        for dirs in test_dir:
            path = os.path.join(self.dirpath, dataset_dir['test'])
            dirpath = os.path.join(path, dirs)
            file_list = glob.glob(os.path.join(dirpath,'*.txt'))

            #ラベル・データを保存するリストを用意
            label_save = []
            data_save = []
            value_save = []
            key_save = []

            #ファイルループ
            file_count = 0
            for filepath in file_list:
                print(" test file (%s) %5d/%5d \r" %(dirs, file_count, len(file_list)), end="")
                file_count += 1
                path,gabage = os.path.splitext(filepath)
                key_save.append(path)

                #テキストからラベルを作成
                test_label = np.zeros(self.label_size)
                with open(path+'.txt','r') as f:
                    tmp = f.read()
                    label_count = 0
                    for label in label_name:
                        index = tmp.find(label)
                        if index != -1:
                            test_label[label_count] = 1
                        label_count += 1
                #リストに変換して保持
                label_save.append(test_label.tolist())
                self.proc(path+'.wav')
                for data in self.proc.processing():
                    #リストに変換して保持
                    data_save.append(data.tolist())
                #リストに変換して保持
                value_save.append(len(self.proc))

            self.dumping(test_dump.format(dirs), data_save, label_save, num=value_save, key=key_save)

    def dumping(self, file, data, label, num=None, key=None):
        dataset = {}
        dataset['data']  = np.array(data, dtype=np.float64)
        dataset['label'] = np.array(label, dtype=np.int32)
        if num is not None:
            dataset['num'] = np.array(num, dtype=np.int32)
            dataset['key'] = key

        save_file = os.path.join(self.dirpath, write_dir)
        save_file = os.path.join(save_file, file)
        with open(save_file, 'wb') as f:
            pickle.dump(dataset, f)

    def sys_dump(self):
        sys = {}

        sys['invide']      = self.proc.invide
        print("Audio invide size : {0}(sec)".format(sys['invide']))
        sys['down_sample'] = self.proc.down_sampling
        print("down sampling     : {0}".format("True" if sys['down_sample'] else "False"))

        sys['ft_win']      = self.proc.win
        print("STFT window size : {0}".format(sys['ft_win']))

        if hasattr(self.proc, 'ch'):
            sys['mfb_ch']  = self.proc.ch
            print("mel-Filter bank channel size : {0}".format(sys['mfb_ch']))
        else:
            sys['mfb_ch']  = None

        sys['org_sample']  = self.proc.sample_rate
        print("Sampling rate of the wave data : {0}(Hz)".format(sys['org_sample']))
        sys['shape']       = self.proc.shape
        print("Data shape : {0}".format(sys['shape']))

        save_file = os.path.join(self.dirpath, write_dir)
        save_file = os.path.join(save_file, setup)
        with open(save_file, 'wb') as f:
            pickle.dump(sys, f)

class Feeder:
    #フィードする
    def __init__(self, path, gpu=False):
        #ダンプファイルがすでにあるかの確認
        self.dump_dir = os.path.join(path,write_dir)
        param_path = os.path.join(self.dump_dir, setup)
        self.gpu = gpu
        if not os.path.exists(param_path):
            Convert(path)()

        self.get_param()


    def get_param(self):
        with open(os.path.join(self.dump_dir, setup), 'rb') as f:
            param_dict = pickle.load(f, encoding='bytes')

        self.shape = param_dict['shape']

    def _load_train_data(self):
        with open(os.path.join(self.dump_dir, train_dump), 'rb') as f:
            dataset = pickle.load(f, encoding='bytes')

        if self.gpu:
            cp = cuda.cupy
            dataset['data']  = cp.array(dataset['data'])
            dataset['label'] = cp.array(dataset['label'])

        shape = (-1,1,)+self.shape
        dataset['data'] = dataset['data'].reshape(shape)

        return dataset

    def _load_test_data(self, dir):
        with open(os.path.join(self.dump_dir, test_dump.format(dir)), 'rb') as f:
            dataset = pickle.load(f, encoding='bytes')

        if self.gpu:
            cp = cuda.cupy
            dataset['data']  = cp.array(dataset['data'])
            dataset['label'] = cp.array(dataset['label'])

        shape = (-1,1,)+self.shape
        dataset['data'] = dataset['data'].reshape(shape)
        dataset['num']  = dataset['num'].tolist()

        return dataset

    def train_feed(self, rand=True, batchsize=100):
        if 'train_dict' not in globals():
            self.train_dict = self._load_train_data()

        data_quantity = self.train_dict['data'].shape[0]
        subscript = np.arange(data_quantity, dtype='int')
        if rand:
            np.random.shuffle(subscript)

        repeat_num = int(data_quantity/batchsize)
        #test 後で消す
        repeat_num = int(repeat_num/2)

        start=0
        end=batchsize
        for n in range(repeat_num):
            if end <= data_quantity:
                ss=subscript[start:end]
            else:
                ss=subscript[start:]
            yield {'data' : self.train_dict['data'][ss], 'label' : self.train_dict['label'][ss], 'size' : batchsize}

            start = end
            end += batchsize

    def test_feed(self, load=0):
        dir_range = np.arange(load+1).tolist()

        #ディレクトリループ
        for dir_num in dir_range:
            test_dict = self._load_test_data(test_dir[dir_num])

            stack = 0
            label_count = 0
            for num in test_dict['num']:
                yield {'data' : test_dict['data'][stack:stack+num], 'label' : test_dict['label'][label_count], 'size' : num}
                stack += num
                label_count += 1
