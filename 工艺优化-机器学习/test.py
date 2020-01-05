# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.model_selection  import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression

data = pd.read_csv("data2.csv")
X = data.iloc[:, 2:]
y = data.iloc[:, 0]

# 数据处理
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
imp = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8','x9', 'x10', 'x11', 'x12','x13', 'x14', 'x15', 'x16','x17', 'x18', 'x19', 'x20', 'x21', 'x22', 'x23', 'x24']
X_train_conti_std = X_train[imp]
X_test_conti_std = X_test[imp]
X_train = pd.DataFrame(data=X_train_conti_std, columns=imp, index=X_train.index)
X_test = pd.DataFrame(data=X_test_conti_std, columns=imp, index=X_test.index)

# 填充缺失项
for column in list(X_train.columns[X_train.isnull().sum() > 0]):
    mean_val = X_train[column].mean()
    X_train[column].fillna(mean_val, inplace=True)

for column in list(X_test.columns[X_test.isnull().sum() > 0]):
    mean_val = X_test[column].mean()
    X_test[column].fillna(mean_val, inplace=True)

# 标准化
scaler1 = preprocessing.StandardScaler().fit(X_train)
X_train = scaler1.transform(X_train)
scaler2 = preprocessing.StandardScaler().fit(X_test)
X_test = scaler2.transform(X_test)

# 多项式回归
poly_reg = PolynomialFeatures(degree=3)
#特征处理
x_train_poly = poly_reg.fit_transform(X_train)
x_test_poly = poly_reg.fit_transform(X_test)
#定义逻辑回归模型
logistic = LogisticRegression(solver='liblinear')
#训练模型
logistic.fit(x_train_poly, y_train)
y_pred = logistic.predict(x_test_poly)

print("多项式系数：")
print(logistic.coef_)

print("")
print('train-score ', logistic.score(x_train_poly, y_train))
print('test-score ', logistic.score(x_test_poly, y_test))



































'''
from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB, BernoulliNB
# 高斯朴素贝叶斯
print("高斯朴素贝叶斯：")
classifier = GaussianNB()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print('Accuracy : {:.2f}'.format(classifier.score(X_test, y_test)))

# 多项分布朴素贝叶斯
print("多项分布朴素贝叶斯：")
classifier = MultinomialNB()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print('Accuracy : {:.2f}'.format(classifier.score(X_test, y_test)))


# 补充朴素贝叶斯
print("补充朴素贝叶斯：")
classifier = ComplementNB()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print('Accuracy : {:.2f}'.format(classifier.score(X_test, y_test)))

# 伯努利朴素贝叶斯
print("伯努利朴素贝叶斯：")
classifier = BernoulliNB()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print('Accuracy : {:.2f}'.format(classifier.score(X_test, y_test)))
'''