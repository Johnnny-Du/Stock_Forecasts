# ENV   : PyCharm
# Time  : 2024/5/13 22:15
# Auther: 杜亚宁
import quandl
import math
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn import model_selection, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import time, datetime

# 获取数据集
# dataset = quandl.get('WIKI/AAPL',start_date = '2003-12-31',end_date = '2023-12-31')
#dataset.to_csv("D:/CUDA_project/stock_forecast/dataset/dataAAPL.csv")
# 读取cvs数据
#dataset = pd.read_csv('D:/CUDA_project/stock_forecast/dataset/AAPL_Data.csv',index_col=0)
# quandl次数超过时使用
dataset = pd.read_csv('D:/CUDA_project/stock_forecast/dataset/dataAAPL.csv',index_col=0)
# 定义预测列变量forecast_col，也就是收盘价
forecast_col = 'Adj. Close'
# 定义预测的天数forecast_out，也就是预测的数据长度，这里设置为所有数据的1%，得到的是一个整数
# math.ceil(x) 返回不小于x的最接近的整数
forecast_out = int(math.ceil(0.01 * len(dataset)))
# 选取只用到的几列 开盘价，最高价，最低价，收盘价，交易额
dataset = dataset[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]

# 构造特征数据HL_PCT、PCT_change
dataset['HL_PCT'] = (dataset['Adj. High'] - dataset['Adj. Low']) / dataset['Adj. Low'] * 100.0
dataset['PCT_change'] = (dataset['Adj. Close'] - dataset['Adj. Open']) / dataset['Adj. Open'] * 100.0

# 选取真正用到的字段
dataset = dataset[[ 'Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]
#print(dataset.head())
# 空值预处理，这里的处理方法是将空值设置为一个比较难出现的值
dataset.fillna(-99999, inplace=True)
# 生成标签列
# 将Adj. Close列数据往前移动1%行，也就是前forecast_out行数据舍掉，剩下后99%的数据往前移动
# shift()函数，用于对dataframe中的数整体上移或下移，当为正数时，向下移。当为负数时，向上移。
dataset['label'] = dataset[forecast_col].shift(-forecast_out)
dataset.to_csv("D:/CUDA_project/stock_forecast/dataset/AAPLDATA.csv")
#dataset.head() #返回列表前n行数据,括号内为空默认返回五行
#print(dataset.head())
#去除label列所有的数据
#dataset.drop(['label'], axis=1)
#print(dataset.drop(['label'], axis=1))

#现在生成真正要用到的模型中过去的数据X、y以及预测时要用到的数据X_lately
X = np.array(dataset.drop(['label'],axis=1))
#数据规范化，使得数据范围在0~1之间
# 将数据按其属性（按列进行）减去其均值，然后除以其方差。最后得到的结果是，对每个属性/每列来说所有数据都聚集在0附近，方差值为1。
X = preprocessing.scale(X)
#将X中后1%行数据作为待预测数据
X_lately = X[-forecast_out:]
#X则取前99%的数据
X = X[:-forecast_out]
#抛弃label列中为空的那些行
dataset.dropna(inplace = True)
#生成对应X的y的数据，即标签值
y = np.array(dataset['label'])
# 机器学习模型训练拟合部分

#将数据分割成训练集和测试集
X_train,X_test,y_train,y_test = model_selection.train_test_split(X,y,test_size = 0.2,random_state=1)

#生成Scikit-Learn的线性回归对象
#这里n_jobs表示使用CPU的个数，当为-1时，代表使用全部的CPU
model = LinearRegression(n_jobs = 4)

#开始训练
model.fit(X_train,y_train)

#评估准确性
accuracy = model.score(X_test,y_test)
print(accuracy)
#进行预测，forecast_set是预测结果
forecast_set = model.predict(X_lately)
# 将预测行结果转化为DataFrame格式的列结果
transet = pd.DataFrame(forecast_set,columns=['Forecast'])
# 获取将要存储的指针位置
start_index = len(dataset)-len(transet)
# dataset中新建列Forecast，用于存放预测结果的数据
dataset['Forecast'] = np.nan
#将预测数据添加到dataset中Forecast列，
dataset.iloc[start_index:, dataset.columns.get_loc('Forecast')] = transet['Forecast'].values
'''
预测部分处理及可视化
'''
#修改matplotlib样式
style.use('ggplot')  #ggplot是背景网格
dataset['Adj. Close'].plot()
dataset['Forecast'].plot()
plt.legend(loc = 'best')
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

df = dataset.iloc[dataset.shape[0]-500-1:dataset.shape[0]-1] #获取后1000行数据，绘制详细图
df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc = 'best')
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()



