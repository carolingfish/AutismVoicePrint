import scipy.io.wavfile
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.fftpack
import scipy
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model.logistic import LogisticRegression
import random
from glob import glob
pd.set_option('display.mpl_style', 'default')

def parseData(filename):
    fo = scipy.io.wavfile.read(filename)
    wave = fo[1][:,0]
    ## draw original waveform
    # plt.plot(range(len(wave)), wave)
    ## draw spetrogram
    # plt.specgram(wave, NFFT=2205, Fs=44100, noverlap=0, cmap=plt.cm.gist_heat)

    ## Number of sample points
    N = 2205  # 44100 / 1000 * 50
    ## sample spacing
    T = 1.0 / 44100.0
    ## time slice is 10ms = 0.01s
    slices = int((len(wave) - N) / 44100.0 / 0.01)
    X = []
    for slice in range(slices):
        # x = np.linspace(0.0, N * T, num=N)
        y = wave[slice * 441: slice * 441 + N]
        # plt.plot(x, y)
        # plt.show(block=True)
        # xf = np.linspace(0.0, 1.0 / (2.0 * T), num=N / 2)
        yf = scipy.fftpack.fft(y)
        ## 5:21 100-400 Hz
        X.append(2.0 / N * np.abs(yf[:N/2])[:21])
        # print xf[np.argmax(2.0/N * np.abs(yf[:N/2]))] ## pitch
        # plt.plot(xf, 2.0/N * np.abs(yf[:N/2]))
        # plt.show(block=True)
    return X


nasal = glob('/Users/lxy/Dropbox/Voice Autism Vocal Samples/Nasalized/*')
normal = glob('/Users/lxy/Dropbox/Voice Autism Vocal Samples/Normal/*')
random.shuffle(nasal)
random.shuffle(normal)
X, y = [], []
for filename in nasal:
    data = parseData(filename)
    X += data
    y += [1] * len(data)
for filename in normal:
    data = parseData(filename)
    X += data
    y += [0] * len(data)

N = len(X) * 9 / 10
X_train, y_train = X[:N], y[:N]
X_test, y_test = X[N:], np.array(y[N:])

# estimator = RandomForestClassifier(n_estimators=200)
estimator = LogisticRegression(class_weight='auto', C=8.0)
estimator.fit(X_train, y_train)
y_predict = estimator.predict(X_test)
print 1.0 * np.sum(y_predict == y_test) / len(y_test)