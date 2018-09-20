import pandas, sys
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import sklearn.metrics as metrics

eng = pd.read_csv("engineered-47k-only-crawled.csv")
noneng = pd.read_csv("non-engineered-only-crawled.csv")

frames = [eng, noneng]

complete = pd.concat(frames)

y = complete.loc[:,'engineered']

complete.drop('engineered', axis=1, inplace=True)

X_train, X_test, y_train, y_test = train_test_split(complete, y, test_size=0.2, stratify=y)

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

clf = svm.SVC()
print("Fitting model...")
clf.fit(X_test, y_test)

print("Making predictions...")
y_pred = clf.predict(X_test)

print('Score:', clf.score(X_test, y_test))
print('Recall (non-engineered):', metrics.recall_score(y_test, y_pred, pos_label=0))
print('Precision (non-engineered):', metrics.recall_score(y_test, y_pred, pos_label=0))
print('Recall (engineered):', metrics.recall_score(y_test, y_pred, pos_label=1))
print('Precision (engineered):', metrics.recall_score(y_test, y_pred, pos_label=1))

TP = 0
FP = 0
TN = 0
FN = 0

yt = y_test.values

for i in range(len(y_pred)):
  if yt[i]==y_pred[i]==1:
    TP += 1
  if y_pred[i]==1 and yt[i]!=y_pred[i]:
    FP += 1
  if yt[i]==y_pred[i]==0:
    TN += 1
  if y_pred[i]==0 and yt[i]!=y_pred[i]:
    FN += 1


print('True-Positives (engineered) = ', TP)
print('False-Positives (engineered) = ', FP)
print('True-Negatives (engineered) = ', TN)
print('False-Negatives (engineered) = ', FN)
