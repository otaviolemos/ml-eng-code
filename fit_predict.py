from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import svm
from xgboost import XGBClassifier
import sklearn.metrics as metrics


def fit_predict(algo, X_train, y_train, X_test, y_test):
    if algo == 'RFC':
        clf = RandomForestClassifier(n_estimators=10)
    elif algo == 'SVM':
        clf = svm.SVC()
    elif algo == 'ABC':
        clf = AdaBoostClassifier();
    elif algo == 'XGB':
        clf = XGBClassifier();
    else:
        print("Algorithm not recognized.")
        return

    print("Fitting model using " + algo + "...")
    clf.fit(X_train, y_train)

    print("Making predictions...")
    y_pred = clf.predict(X_test)

    print(algo + " results:")
    print('Score:', clf.score(X_test, y_test))
    print('Recall (non-engineered):', metrics.recall_score(y_test, y_pred, pos_label=0))
    print('Precision (non-engineered):', metrics.precision_score(y_test, y_pred, pos_label=0))
    print('Recall (engineered):', metrics.recall_score(y_test, y_pred, pos_label=1))
    print('Precision (engineered):', metrics.precision_score(y_test, y_pred, pos_label=1))
