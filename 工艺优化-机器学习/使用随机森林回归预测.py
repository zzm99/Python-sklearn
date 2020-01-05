# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection  import train_test_split

# 读取数据并查看:
data = pd.read_csv("data2.csv")

# 数据集基本行列都确定之后，我们就可以进行分割了，这里将30%的数据集作为测试集：
X = data.iloc[:, 1:]
y = data.iloc[:, 0]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# 这里只取几个重点参数
imp = ['x4', 'x5', 'x6', 'x13', 'x16', 'x17', 'x18', 'x20']
X_train_conti_std = X_train[imp]
X_test_conti_std = X_test[imp]
# 将ndarray转为dataframe
X_train = pd.DataFrame(data=X_train_conti_std, columns=imp, index=X_train.index)
X_test = pd.DataFrame(data=X_test_conti_std, columns=imp, index=X_test.index)

rf_regressor=RandomForestRegressor(max_depth=11,n_estimators=120)
rf_regressor.fit(X_train,y_train)

# 在测试集上的准确率
print('Score: {:.2f}'.format(rf_regressor.score(X_test, y_test)))

