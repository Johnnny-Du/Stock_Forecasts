# coding:utf-8
# author:杜亚宁
# Time:2024/5/17 18:37
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import datetime
import matplotlib.pyplot as plt
from matplotlib import style

# 读取数据
dataset = pd.read_csv('D:/CUDA_project/stock_forecast/dataset/dataAAPL.csv', index_col=0)

# 选择主要参数
dataset = dataset[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]

# 创建新特征
dataset['HL_PCT'] = (dataset['Adj. High'] - dataset['Adj. Low']) / dataset['Adj. Low'] * 100.0
dataset['PCT_change'] = (dataset['Adj. Close'] - dataset['Adj. Open']) / dataset['Adj. Open'] * 100.0

# 选择预测的列
forecast_col = 'Adj. Close'

# 填充缺失值
dataset.fillna(value=-99999, inplace=True)

# 预测未来多少天
forecast_out = int(np.ceil(0.01 * len(dataset)))

# 创建标签列
dataset['label'] = dataset[forecast_col].shift(-forecast_out)

# 特征集和标签集
X = np.array(dataset.drop(['label'], axis=1))
X = StandardScaler().fit_transform(X)

# 预测集
X_lately = X[-forecast_out:]
X = X[:-forecast_out]

# 删除缺失值的标签行
dataset.dropna(inplace=True)
y = np.array(dataset['label'])

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 创建线性回归模型
model = LinearRegression()

# 训练模型
model.fit(X_train, y_train)

# 测试模型
accuracy = model.score(X_test, y_test)
print(f'Accuracy: {accuracy}')

# 预测未来的数据
forecast_set = model.predict(X_lately)

# 将预测结果转化为DataFrame格式的列结果
transet = pd.DataFrame(forecast_set, columns=['Forecast'])

# 在dataset中新建列Forecast，并填充NaN
dataset['Forecast'] = np.nan

# 确定起始位置并赋值预测结果
last_date = dataset.index[-1]
last_unix = datetime.datetime.strptime(last_date, '%Y-%m-%d').timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix).strftime('%Y-%m-%d')
    next_unix += one_day
    dataset.loc[next_date] = [np.nan for _ in range(len(dataset.columns)-1)] + [i]

# 查看结果
print(dataset.tail(10))



style.use('ggplot')

dataset['Adj. Close'].plot()
dataset['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
