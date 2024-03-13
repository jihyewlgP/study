from sklearn.metrics import mean_squared_error
import numpy as np
import random
import os
import matplotlib.pyplot as plt
import seaborn as sns

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

seed_everything(42) # Seed 고정

import pandas as pd

train = pd.read_csv('D://Dev//dacon//open//train.csv')
test = pd.read_csv('D://Dev//dacon//open//test.csv')

train.info()

# 타겟 변수인 'y'의 분포를 시각화합니다.
plt.figure(figsize=(10, 6))
sns.histplot(train['Income'], bins=30, kde=True)
plt.title('Distribution of y')
plt.show()

train_x = train.drop(columns=['ID', 'Income'])
valid_x = train.drop(columns=['ID', 'Income'])
train_y = train['Income']

plt.figure(figsize=(16, 12))
sns.heatmap(train.corr(), annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# =============================================================================
#train_x = train_x[:14000]
#train_y = train['Income'][:14000]

#valid_x = valid_x[14000:]
#valid_y = train['Income'][14000:]

test_x = test.drop(columns=['ID'])
# 
# =============================================================================
from sklearn.preprocessing import LabelEncoder

encoding_target = list(train_x.dtypes[train_x.dtypes == "object"].index)

for i in encoding_target:
    le = LabelEncoder()
    
    # train과 test 데이터셋에서 해당 열의 모든 값을 문자열로 변환
    train_x[i] = train_x[i].astype(str)
    #valid_x[i] = valid_x[i].astype(str)
    test_x[i] = test_x[i].astype(str)
    
    le.fit(train_x[i])
    train_x[i] = le.transform(train_x[i])
    
# =============================================================================
#     for case in np.unique(valid_x[i]):
#         if case not in le.classes_: 
#             le.classes_ = np.append(le.classes_, case)
#  
#     valid_x[i] = le.transform(valid_x[i])
# =============================================================================

    # test 데이터의 새로운 카테고리에 대해 le.classes_ 배열에 추가
    for case in np.unique(test_x[i]):
        if case not in le.classes_: 
            le.classes_ = np.append(le.classes_, case)
    
    test_x[i] = le.transform(test_x[i])
    
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
#from catboost import CatBoostRegressor
import xgboost
from sklearn.metrics import explained_variance_score
from sklearn.svm import SVR
from lightgbm import LGBMRegressor, early_stopping
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV

lgbm = LGBMRegressor(n_estimators=408, learning_rate=0.01, max_depth=9)

lgbm.fit(train_x, train_y)
score = lgbm.score(train_x, train_y)
y_pred = lgbm.predict(test_x)

submission = pd.read_csv('D://Dev//dacon//open//sample_submission.csv')
submission['Income'] = y_pred
submission

submission.to_csv('D://Dev//dacon//open//two_submission.csv', index=False)

#440(408) 0.01,9
# 조기종료 훈련, 검증데이터셋
evals=[(valid_x, valid_y)]
# fit  50번이상 성능개선 안되면 종료, 평가지표는 logloss 
lgbm.fit(train_x, train_y, eval_set=evals, callbacks=[early_stopping(stopping_rounds=50)])

#lgbm.fit(train_x, train_y)
y_pred2 = lgbm.predict(valid_x)
print(mean_squared_error(valid_y, y_pred, squared=False))
#561.6376010000113