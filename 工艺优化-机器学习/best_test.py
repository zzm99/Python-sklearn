# -*- coding: utf-8 -*-
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, classification_report
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression

# 读取数据
data = pd.read_csv("data2.csv")
X = data.iloc[:, 2:]
y = data.iloc[:, 0]

# 数据处理
#imp = ['x4','x5','x6','x13','x16','x17','x18','x20']
imp = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8','x9', 
       'x10','x11', 'x12','x13', 'x14', 'x15', 'x16','x17', 
       'x18','x19', 'x20', 'x21', 'x22', 'x23', 'x24']
X_train_conti_std = X[imp]

X_train = pd.DataFrame(data=X_train_conti_std, columns=imp)

# 填充缺失项
for column in list(X_train.columns[X_train.isnull().sum() > 0]):
    mean_val = X_train[column].mean()
    X_train[column].fillna(mean_val, inplace=True)
    
# 标准化
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)

# 多项式回归
poly_reg = PolynomialFeatures(degree=3, interaction_only=True)

# 特征处理
x_train_poly = poly_reg.fit_transform(X_train)

# 定义逻辑回归模型
logistic = LogisticRegression(solver='liblinear')

# 训练模型
logistic.fit(x_train_poly, y)
y_pred = logistic.predict(x_train_poly)

print("-------------------------------------------")
print("多项式系数：")
print(logistic.coef_)
print("-------------------------------------------")
print('score: ', logistic.score(x_train_poly, y))
print('accuracy_score: ', accuracy_score(y, y_pred)) 
print("-------------------------------------------")
print("混淆矩阵")
cm = confusion_matrix(y, y_pred)
print(cm)
print("-------------------------------------------")
print('precision_score: ', precision_score(y, y_pred)) 
print("-------------------------------------------")
print('classification_report: ')
print(classification_report(y, y_pred)) 
print("-------------------------------------------")


