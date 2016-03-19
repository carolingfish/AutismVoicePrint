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
import VoiceFilter_ck as vf
import matplotlib
pd.set_option('display.mpl_style', 'default')
matplotlib.style.use('ggplot')

def parseData(filename):
    # fo = scipy.io.wavfile.read(filename)
    # wave = fo[1][:,0]
    
    #Use caroling's filter
    sdata = vf.GetSoundFile(filename)
    sdata2 = vf.FilterLowEnergy(sdata)
    sdata3 = vf.PitchFilter(sdata2)
    # print(len(sdata))
    # print(len(sdata2))
    # print(len(sdata3))
    wave = sdata3
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
def combime_data(files,nasality):
    X, y = [], []
    for filename in files:
        data = parseData(filename)
        X += data
        y += [1 if (nasality==True) else 0] * len(data)
    return X,y

# def train_valid_test_split(data,train_p,valid_p,test_p):


nasal = glob('/Users/Eric/Dropbox/Voice Autism Vocal Samples/train/Nasalized/*')
normal = glob('/Users/Eric/Dropbox/Voice Autism Vocal Samples/train/Normal/*')
nasal_test = glob('/Users/Eric/Dropbox/Voice Autism Vocal Samples/test/Nasalized/*')
normal_test = glob('/Users/Eric/Dropbox/Voice Autism Vocal Samples/test/Normal/*')
 

X_nasal,y_nasal = combime_data(nasal,True)
X_normal,y_normal = combime_data(normal,False)

X = X_nasal + X_normal
y = y_nasal + y_normal
index_shuf = list(range(len(X)))
random.shuffle(index_shuf)

X = [X[i] for i in index_shuf]
y = [y[i] for i in index_shuf]
# quit()

N_train = int(len(X_nasal)*7/10)

X_train, y_train = X[:N_train], y[:N_train]
X_valid, y_valid = X[N_train:],y[N_train:]

Cs = np.logspace(-2, 5, 10)
valid_predict = []

for C in Cs:
    estimator = LogisticRegression(class_weight='auto', C=C)
    estimator.fit(X_train, y_train)
    y_predict_val = estimator.predict(X_valid)
    valid_predict.append(1.0 * np.sum(y_predict_val == y_valid) / len(y_valid))

valid_predict = np.array(valid_predict)
C = Cs[np.argmax(valid_predict)]

print("C:",C, "Accurary(valid):", np.max(valid_predict))
# estimator = RandomForestClassifier(n_estimators=200)


estimator = LogisticRegression(class_weight='auto', C=C)
estimator.fit(X, y)



for file in normal_test:
    X_test,y_test = combime_data([file],False)
    y_predict = estimator.predict(X_test)
    # print(X_nasal_test)

    print ("NORMAL__Accurary(test_nasal):",1.0 * np.sum(y_predict == y_test) / len(y_test))

    false_nasal = 0
    false_normal = 0

    for i in range(len(y_test)):

        if y_predict[i]==1 and y_test[i]==0:
            false_nasal += 1
        elif y_predict[i]==0 and y_test[i]==1:
            false_normal += 1
    # print("NORMAL__False Nasal: ",false_nasal/len(y_test))
    # print("NORMAL__False Normal: ",false_normal/len(y_test))

# print(Cs)
# print(valid_predict)
# plt.plot(Cs,valid_predict)
# plt.show()
for file in nasal_test:
    X_test,y_test = combime_data([file],True)
    print(X_test)
    y_predict = estimator.predict(X_test)
    # print(X_nasal_test)

    print ("NASAL__Accurary(test_nasal):",1.0 * np.sum(y_predict == y_test) / len(y_test))

    false_nasal = 0
    false_normal = 0

    for i in range(len(y_test)):

        if y_predict[i]==1 and y_test[i]==0:
            false_nasal += 1
        elif y_predict[i]==0 and y_test[i]==1:
            false_normal += 1
    # print("NASAL__False Nasal: ",false_nasal/len(y_test))
    # print("NASAL__False Normal: ",false_normal/len(y_test))

