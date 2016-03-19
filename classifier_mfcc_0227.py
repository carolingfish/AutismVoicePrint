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
import matplotlib
pd.set_option('display.mpl_style', 'default')
matplotlib.style.use('ggplot')

random.seed(100)

X,y = [],[]
# Nasal
data = np.loadtxt('nasal_MFCCs.csv',delimiter=',') 
X = data
y += [1] * len(data)

# Normal
data = np.loadtxt('normal_MFCCs.csv',delimiter=',') 
X = np.concatenate((X, data), axis=0)
y += [0] * len(data)

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
print ("Accurary(test):",1.0 * np.sum(y_predict == y_test) / len(y_test))

false_nasal = 0
false_normal = 0

for i in range(len(y_test)):
    if y_predict[i]==1 and y_test[i]==0:
        false_nasal += 1
    elif y_predict[i]==0 and y_test[i]==1:
        false_normal += 1
print("False Nasal: ",false_nasal/len(y_test))
print("False Normal: ",false_normal/len(y_test))

# print(Cs)
# print(valid_predict)
plt.plot(Cs,valid_predict)
plt.show()