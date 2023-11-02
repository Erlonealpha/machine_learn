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

#查看模型相关特征
print('各特征的相关系数为：\n', lr_model.coef_)

print('模型的截距为：', lr_model.intercept_)

print('模型的迭代次数为：', lr_model.n_iter_)

print('预测测试集前10个结果为：\n', lr_model.predict(x_testStd)[: 10])

print('测试集准确率为：', lr_model.score(x_testStd, y_test))

print('测试集前3个对应类别的概率为：\n', lr_model.predict_proba(x_testStd)[: 3])

print('测试集前3个对应类别的概率的log值为：\n',
      lr_model.predict_log_proba(x_testStd)[: 3])

print('测试集前3个的决策函数值为：\n',
      lr_model.decision_function(x_testStd)[: 3])

print('模型的参数为：\n', lr_model.get_params())

print('修改max_iter参数为1000后的模型为：\n', lr_model.set_params(max_iter=1000))

print('系数矩阵转为密度数组后的模型为：\n', lr_model.densify())

print('系数矩阵转为稀疏形式后的模型为：\n', lr_model.sparsify())
