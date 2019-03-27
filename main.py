import pandas as pd
from sklearn.model_selection import train_test_split
from fit_predict import fit_predict


eng = pd.read_csv("/Users/otaviolemos/Dropbox/academic/projects/eng-vs-noneng/results/metrics-data/github-550-eng/mlcc_input.file")
noneng = pd.read_csv("/Users/otaviolemos/Dropbox/academic/projects/eng-vs-noneng/results/metrics-data/github-10k-neng/mlcc_input.file")

eng['engineered'] = 1
noneng['engineered'] = 0

eng.drop('EMC', axis=1, inplace=True)
eng.drop('EXH', axis=1, inplace=True)
eng.drop('UNAND', axis=1, inplace=True)
eng.drop('UNPOR', axis=1, inplace=True)
eng.drop('HEFF', axis=1, inplace=True)
eng.drop('HVOL', axis=1, inplace=True)
eng.drop('name', axis=1, inplace=True)
eng.drop('path', axis=1, inplace=True)

noneng.drop('EXH', axis=1, inplace=True)
noneng.drop('UNAND', axis=1, inplace=True)
noneng.drop('UNPOR', axis=1, inplace=True)
noneng.drop('HEFF', axis=1, inplace=True)
noneng.drop('HVOL', axis=1, inplace=True)
noneng.drop('name', axis=1, inplace=True)
noneng.drop('path', axis=1, inplace=True)

noneng = noneng[noneng.SLOC > 2] #se quiser utilizar somente métodos com SLOC > 2
eng = eng[eng.SLOC > 2]

engSample = eng.sample(n=len(noneng));

frames = [engSample, noneng]

complete = pd.concat(frames)
complete = complete.dropna()

y = complete.loc[:,'engineered']

complete = complete.drop(complete.columns[0], axis=1) # removing method name and path columns
complete = complete.drop(complete.columns[0], axis=1)
complete.drop('engineered', axis=1, inplace=True)     # removing engineered column

X_train, X_test, y_train, y_test = train_test_split(complete, y, test_size=0.2)

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

fit_predict("RFC", X_train, y_train, X_test, y_test)
