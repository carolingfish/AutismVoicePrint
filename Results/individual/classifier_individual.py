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
import csv
import pdb
pd.set_option('display.mpl_style', 'default')
matplotlib.style.use('ggplot')

def parseData(filename):
    # fo = scipy.io.wavfile.read(filename)
    # wave = fo[1][:,0]
    #pdb.set_trace()
    
    #Use caroling's filter
    sdata = vf.GetSoundFile(filename)
    sdata2 = vf.FilterLowEnergy(sdata)
    sdata3 = vf.PitchFilter(sdata2)
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

#name = "F_CK"
#name = "F_SY"
#name = "F_XY"
#name = "M_FC"
name = "M_EL"

nasal = glob('/Users/feichao/Dropbox/Vocal Samples/Nasal/'+name+'*')
normal = glob('/Users/feichao/Dropbox/Vocal Samples/Normal/'+name+'*')
#nasal = glob('/Users/feichao/Downloads/Nasal/*')
#normal = glob('/Users/feichao/Downloads/Normal/*')

#pdb.set_trace()
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

index_shuf = list(range(len(X)))
random.shuffle(index_shuf)

X = [X[i] for i in index_shuf]
y = [y[i] for i in index_shuf]

#pdb.set_trace()
N = int(len(X) * 9 / 10)
X_train, y_train = X[:N], y[:N]
X_test, y_test = X[N:], np.array(y[N:])


N_train = int(len(X)*7/10)
N_valid = int(len(X)*9/10)
X_train, y_train = X[:N_train], y[:N_train]
X_valid, y_valid = X[N_train:N_valid],y[N_train:N_valid]
X_test, y_test = X[N_valid:], np.array(y[N_valid:])

Cs = np.logspace(-2, 5, 50)#Inverse of regularization strength;
valid_predict = []

#pdb.set_trace()
for C in Cs:
    estimator = LogisticRegression(class_weight='auto', C=C)
    estimator.fit(X_train, y_train)
    y_predict_val = estimator.predict(X_valid)
    valid_predict.append(1.0 * np.sum(y_predict_val == y_valid) / len(y_valid))
#pdb.set_trace()

valid_predict = np.array(valid_predict)
C = Cs[np.argmax(valid_predict)]

print("C:",C, "Accurary(valid):", np.max(valid_predict))
# estimator = RandomForestClassifier(n_estimators=200)


estimator = LogisticRegression(class_weight='auto', C=C)
estimator.fit(X_train, y_train)
y_predict = estimator.predict(X_test)
acc = 1.0 * np.sum(y_predict == y_test) / len(y_test)
print ("Accurary(test):",acc)

false_nasal = 0
false_normal = 0

#pdb.set_trace()
for i in range(len(y_test)):
    if y_predict[i]==1 and y_test[i]==0:
        false_nasal += 1
    elif y_predict[i]==0 and y_test[i]==1:
        false_normal += 1
print("False Nasal: ",float(false_nasal)/len(y_test))
print("False Normal: ",float(false_normal)/len(y_test))
#pdb.set_trace()
#print(Cs)
#print(valid_predict)

with open(name+".csv", "w") as text_file:
            text_file.write("Accurary(test):{0}\n".format(str(acc)))

with open(name+".csv", "a") as text_file:
            text_file.write("False Nasal:{0}\n".format(str(float(false_nasal)/len(y_test))))

with open(name+".csv", "a") as text_file:
            text_file.write("False Normal:{0}\n".format(str(float(false_normal)/len(y_test))))

with open(name+".csv", 'a') as f:
    writer = csv.writer(f, delimiter='\t')
    writer.writerows(zip(Cs, valid_predict))


#plt.plot(Cs,valid_predict)
#plt.show()



