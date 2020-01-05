# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.model_selection  import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

# 读取数据并查看:
data = pd.read_csv("data2.csv")

# 选取数据集中有用的特征:
# drop_arr = ['x5', 'x6', 'x7', 'x8','x9', 'x10', 'x11', 'x12','x13', 'x14', 'x15', 'x16','x17', 'x18', 'x19', 'x20', 'x21', 'x22', 'x23', 'x24']
# data = data.drop(labels=drop_arr, axis=1)

# 数据集基本行列都确定之后，我们就可以进行分割了，这里将30%的数据集作为测试集：
X = data.iloc[:, 2:]
y = data.iloc[:, 0]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# 标准化:
# 这里只取几个重点参数
# imp = ['x4', 'x5', 'x6', 'x13', 'x16', 'x17', 'x18', 'x20']
imp = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8','x9', 'x11', 'x12','x13', 'x14', 'x15', 'x16','x17', 'x18', 'x19', 'x20', 'x21', 'x22', 'x23', 'x24']
X_train_conti_std = X_train[imp]
X_test_conti_std = X_test[imp]
# 将ndarray转为dataframe
X_train = pd.DataFrame(data=X_train_conti_std, columns=imp, index=X_train.index)
X_test = pd.DataFrame(data=X_test_conti_std, columns=imp, index=X_test.index)

# 使用sklearn中逻辑回归模型进行训练并做预测:
# 基于训练集使用逻辑回归建模
classifier = LogisticRegression()
classifier.fit(X_train, y_train)
# print(classifier.coef_) # 权重

# 将模型应用于测试集并查看混淆矩阵
y_pred = classifier.predict(X_test)
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)
# 在测试集上的准确率
print('Score: {:.2f}'.format(classifier.score(X_test, y_test)))
