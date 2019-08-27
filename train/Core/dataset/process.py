# coding: utf-8
#chainer

import glob
import numpy as np
from scipy import fromstring, int16
from scipy.fftpack import fft

class Chank:
    def __init__(self, x, win):
        self.hop = int(win/2)
        self.audio = x[:int(x.size/self.hop)*self.hop].reshape((-1,self.hop))

    def __len__(self):
        return len(self.audio)

    def feed(self):
        n = 2
        while(1):
            if n > len(self.audio):
                raise StopIteration

            data = self.audio[n-2:n].reshape(-1)
            n += 1
            yield data

#stft
def stft(x, win):
    hop = int(win/2)
    audio = Chank(x,win)
    spec = np.zeros(len(audio)*hop).reshape((-1,hop))

    n = 0
    for chank in audio.feed():
        F = chank * np.hamming(win)
        X = np.fft.fft(F, win)
        spec[n] = np.abs(X)[:hop]
        n += 1

    return spec

#メルフィルタバンク
def hz2mel(f):
    #Hzをmelに変換
    return 1127.01048 * np.log(f / 700.0 + 1.0)

def mel2hz(m):
    #melをhzに変換
    return 700.0 * (np.exp(m / 1127.01048) - 1.0)

def mfb(fs, win, ch):
    fmax = fs/2 #ナイキスト周波数(Hz)
    melmax = hz2mel(fmax) #ナイキスト周波数（ml）
    nmax = (int)(win/2) #周波数インデックスの最大数
    df = fs/win #周波数解像度(周波数インデックス１あたりのHz幅)
    dmel = melmax / (ch+1) #メル尺度における各フィルタの中心周波数を求める
    melcenters = np.arange(1, ch+1)*dmel

    fcenters = mel2hz(melcenters) #各フィルタの中心周波数をHzに変換
    indexcenter = fcenters / df #各フィルタの中心周波数を周波数インデックスに変換
    indexstart = np.hstack(([0], indexcenter[0:ch-1])) #各フィルタの開始位置のインデックス
    indexstop = np.hstack((indexcenter[1:ch], [nmax])) #各フィルタの終了位置のインデックス

    filterbank = np.zeros((ch,nmax))

    for c in np.arange(0, ch):
        s=int(indexstart[c])
        m=int(indexcenter[c])
        e=int(indexstop[c])
        #三角フィルタの左の直線の傾きから点を求める
        increment = 1.0 / (indexcenter[c] - indexstart[c])
        for i in np.arange(s, m):
            filterbank[c][i] = (i - indexstart[c]) * increment

        #三角フィルタの右の直線の傾きから点を求める
        decrement = 1.0 / (indexstop[c] - indexcenter[c])
        for i in np.arange(m, e):
            filterbank[c][i] = 1.0 - ((i - indexcenter[c]) * decrement)


        min=np.min(filterbank[c])
        max=np.max(filterbank[c])
        filterbank[c][s:e] = (filterbank[c][s:e]-min)/(max-min)

    return filterbank

class AudioProcess:
    def __init__(self, wave_pram, ft_pram, mfb_pram = None):
        self.invide        = wave_pram[0] #秒
        self.down_sampling = wave_pram[1] #T/F

        self.win = ft_pram[0]

        if mfb_pram is None :
            self.non_mfb = True
        else :
            self.mfb = None
            self.ch = mfb_pram[0]
            self.non_mfb = False

    def wav_import(self, path):
        import wave
        with wave.open(path, 'rb') as wd:
            tmp = wd.readframes(wd.getnframes())
            num_data = fromstring(tmp, dtype = int16)

            #モノラル化
            if (wd.getnchannels() == 2):
                # 左チャンネル
                left = num_data[::2]
                # 右チャンネル
                right = num_data[1::2]
                wav_data_tmp = left + right
            else :
                wav_data_tmp = num_data

            #ダウンサンプリング
            if self.down_sampling:
                self.sample_rate = int(wd.getframerate()/2)
                audio = wav_data_tmp[::2]
            else:
                self.sample_rate = int(wd.getframerate())
                audio = wav_data_tmp

        return np.array(audio)

    def __call__(self, path):
        audio = self.wav_import(path)
        if not self.non_mfb:
            if self.mfb is None:
                self.mfb = mfb(self.sample_rate, self.win, self.ch)

        hop = int(self.sample_rate/2)
        if audio.size%self.sample_rate != 0:
            pad = np.zeros(self.sample_rate-(audio.size%self.sample_rate))
            data = np.concatenate((audio, pad))
        else:
            data = audio
        self.invide_data = data.reshape((-1,hop))

    def processing(self):
        n = 2
        while(1):
            if n > len(self.invide_data):
                raise StopIteration

            data = self.invide_data[n-2:n].reshape(-1)

            #stft
            data = stft(data,self.win)

            if not self.non_mfb:
                # 振幅スペクトルにメルフィルタバンクを適用
                data = np.dot(data, self.mfb.T)+1
                data = np.log10(data)

            # 平坦化
            self.shape = data.shape
            flat_data = data.reshape((-1))
            n += 1
            yield flat_data

    def __len__(self):
        return len(self.invide_data)-1
