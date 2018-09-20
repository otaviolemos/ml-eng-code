import pandas as pd
from sklearn.model_selection import train_test_split
from fit_predict import fit_predict


eng = pd.read_csv("/Users/otaviolemos/eclipse-workspace/java-parser/github-550_metric_output/mlcc_input.file")
noneng = pd.read_csv("/Users/otaviolemos/eclipse-workspace/java-parser/github-550-ne_metric_output/mlcc_input.file")

eng['engineered'] = 1
noneng['engineered'] = 0

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
