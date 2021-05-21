import pandas as pd
import numpy as np
from IPython.display import display
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

#keep all outputs, not only the last one
from IPython.core.interactiveshell import InteractiveShell 
InteractiveShell.ast_node_interactivity = 'all'

#data=pd.read_csv("./adult.data", header=None, index_col=False)
#data.shape
data = pd.read_csv(
    "./adult.data", header=None, index_col=False,
    names=['age', 'workclass', 'fnlwgt', 'education',  'education-num',
           'marital-status', 'occupation', 'relationship', 'race', 'gender',
           'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
           'income'])

#display(data[:5])
#data[9].value_counts()
#data.columns
data_dummies=pd.get_dummies(data)
#list(data_dummies.columns)
data_dummies[:21]
# arr=np.loadtxt("./adult.data")
# arr.shape

#使用lable来获取dataframe的一部分并作为array
#冒号切片，第一个冒号表示获取的行，第二个冒号表示获取的列

# p=data_dummies.columns
# p=np.array(p)
# p
# data_dummies.head()

features = data_dummies.loc[:,'age':'occupation_ Transport-moving'] 

X=features.values

y=data_dummies['income_ >50K'].values

X.shape
y.shape

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.2)
logreg=LogisticRegression()
logreg.fit(X_train, y_train)
print(logreg.score(X_test, y_test))

#features

# print(features)
