from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
'''
分类模型,将load_breast_cancer数据集划分为训练集和测试集
'''

cancer = load_breast_cancer()
x=cancer['data']
y=cancer['target']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=22)
print('x_train第一行数据为: \n',X_train[0:1],'\n','y_train第一个数据为: ',y_train[0:1])

'''
构建logistic回归模型并训练模型
'''

from sklearn.preprocessing import StandardScaler
stdScaler = StandardScaler().fit(X_train)
x_trainStd = stdScaler.transform(X_train)
x_testStd = stdScaler.transform(X_test)
# 使用LogisticRegression类构建Logistic回归模型
from sklearn.linear_model import LogisticRegression
lr_model = LogisticRegression(solver='saga')
# 训练Logistic回归模型
lr_model.fit(x_trainStd, y_train)
print('训练出来的LogisticRegression模型为: \n', lr_model)