# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np 
from sklearn.model_selection  import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

data = pd.read_csv("exscores.CSV")

# 将30%的数据集作为测试集：
X = data.iloc[:, 1:]
y = data.iloc[:, 5]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

imp = ['x1', 'x2', 'x3', 'x4']
X_train_conti_std = X_train[imp]
X_test_conti_std = X_test[imp]

X_train = pd.DataFrame(data=X_train_conti_std, columns=imp, index=X_train.index)
X_test = pd.DataFrame(data=X_test_conti_std, columns=imp, index=X_test.index)

classifier = LinearRegression()
classifier.fit(X_train, y_train)

# print(classifier.coef_) # 权重

y_pred = classifier.predict(X_test)

# print('Score: {:.2f}'.format(classifier.score(X_test, y_pred))) # 测试集精确度

MSE = metrics.mean_squared_error(y_test, y_pred)
RMSE = np.sqrt(metrics.mean_squared_error(y_test, y_pred))

print('MSE:',MSE)
print('RMSE:',RMSE)