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
#import VoiceFilter_ck as vf
import matplotlib
import csv
import pdb
pd.set_option('display.mpl_style', 'default')
matplotlib.style.use('ggplot')
#pdb.set_trace()
X,y = [],[]
dataset = "F_MFCCs"
# Nasal
data = np.loadtxt('nasal_'+dataset+'.csv',delimiter=',') 
X = data
y += [1] * len(data)

# Normal
data = np.loadtxt('normal_'+dataset+'.csv',delimiter=',') 
X = np.concatenate((X,data),axis=0)
#X += data
y += [0] * len(data)

random.seed(100)
index_shuf = list(range(len(X)))
random.shuffle(index_shuf)
X = [X[i] for i in index_shuf]
y = [y[i] for i in index_shuf]
# N = int(len(X) * 9 / 10)
# X_train, y_train = X[:N], y[:N]
# X_test, y_test = X[N:], np.array(y[N:])


N_train = int(len(X)*6/10)
N_valid = int(len(X)*8/10)
X_train, y_train = X[:N_train], y[:N_train]
X_valid, y_valid = X[N_train:N_valid],y[N_train:N_valid]
X_test, y_test = X[N_valid:], np.array(y[N_valid:])

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
estimator.fit(X_train, y_train)
y_predict = estimator.predict(X_test)
print(y_predict)
print(y_test)
acc = 1.0 * np.sum(y_predict == y_test) / len(y_test)
print ("Accurary(test):",acc)

false_nasal = 0
false_normal = 0

for i in range(len(y_test)):
    if y_predict[i]==1 and y_test[i]==0:
        false_nasal += 1
    elif y_predict[i]==0 and y_test[i]==1:
        false_normal += 1
print("False Nasal: ",float(false_nasal)/len(y_test))
print("False Normal: ",float(false_normal)/len(y_test))

with open(dataset+".csv", "w") as text_file:
            text_file.write("Accurary(test):{0}\n".format(str(acc)))

with open(dataset+".csv", "a") as text_file:
            text_file.write("False Nasal:{0}\n".format(str(float(false_nasal)/len(y_test))))

with open(dataset+".csv", "a") as text_file:
            text_file.write("False Normal:{0}\n".format(str(float(false_normal)/len(y_test))))

with open(dataset+".csv", 'a') as f:
    writer = csv.writer(f, delimiter='\t')
    writer.writerows(zip(Cs, valid_predict))

# print(Cs)
# print(valid_predict)
#plt.plot(Cs,valid_predict)
#plt.show()