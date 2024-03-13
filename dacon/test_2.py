import pandas as pd
import keras
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
# LSTM 의 경우 2차원의 데이터를 쓸때 필요, 그게 아닐경우 필요없음.

data = pd.read_csv('S://개인업무문서//1_데이터분석 & 코드 & 자료//seasonal_decompose.csv')
data2 = pd.read_csv('D://Dev//dacon//open//train.csv')
X= data['date']
y= data['price']

# 임의의 1차원 시계열 데이터를 가정합니다.
# 이 예제에서는 pandas Series를 사용하며, 인덱스는 날짜, 값은 관측치를 나타냅니다.

# ARIMA 모델 정의
# 여기서는 AR 차수(p)를 5, 차분(d)을 1, MA 차수(q)를 0으로 설정합니다.
model = ARIMA(y, order=(8, 1, 0))

# 모델 학습
model_fit = model.fit()

# 예측 수행
# 이 예제에서는 마지막 10개의 관측치를 예측합니다.
forecast, conf_int = model_fit.forecast(steps=10)

print('Forecast: ', forecast)


model = Sequential()

#Adding the first LSTM layer and some Dropout regularisation
model.add(LSTM(50, return_sequences = True, input_shape = (X.shape[0], 1)))
model.add(Dropout(0.2))
# Adding a second LSTM layer and some Dropout regularisation
model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))
# Adding a third LSTM layer and some Dropout regularisation
model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))
# Adding a fourth LSTM layer and some Dropout regularisation
model.add(LSTM(50))
# model.add(Dropout(0.2))
# Adding the output layer
model.add(Dense(units = 1))
model.add(Dense(20, activation='softmax'))

# Compiling the RNN
model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics=['accuracy'])

# Fitting the RNN to the Training set
model.fit(X, y, epochs = 10, batch_size = 32)


#==========================
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

display(train.head(3))
display(test.head(3))

train.info()

# 타겟 변수인 'y'의 분포를 시각화합니다.
plt.figure(figsize=(10, 6))
sns.histplot(train['Income'], bins=30, kde=True)
plt.title('Distribution of y')
plt.show()

train_x = train.drop(columns=['ID', 'Income'])
valid_x = train.drop(columns=['ID', 'Income'])
train_y = train['Income']
# 각 변수간의 상관관계를 시각화합니다.
#data = pd.concat([train_x, train_y], axis = 1 )
#data_2 = data[['Working_Week (Yearly)', 'Occupation_Status', 'Martial_Status', 'Household_Status', 'Household_Summary', 'Tax_Status', 'Income']]

#train_x = data_2.drop(columns='Income')
#valid_x = data_2.drop(columns='Income')

plt.figure(figsize=(16, 12))
sns.heatmap(data_2.corr(), annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# =============================================================================
# train_x = train_x[:14000]
# train_y = train['Income'][:14000]
# 
# valid_x = valid_x[14000:]
# valid_y = train['Income'][14000:]
# 
# test_x = test.drop(columns=['ID'])
# 
# =============================================================================
from sklearn.preprocessing import LabelEncoder

encoding_target = list(train_x.dtypes[train_x.dtypes == "object"].index)
#encoding_target_valid = list(valid_x.dtypes[valid_x.dtypes == "object"].index)

for i in encoding_target:
    le = LabelEncoder()
    
    # train과 test 데이터셋에서 해당 열의 모든 값을 문자열로 변환
    train_x[i] = train_x[i].astype(str)
    #valid_x[i] = valid_x[i].astype(str)
    test_x[i] = test_x[i].astype(str)
    
    le.fit(train_x[i])
    train_x[i] = le.transform(train_x[i])
    
    #for case in np.unique(valid_x[i]):
    #    if case not in le.classes_: 
    #        le.classes_ = np.append(le.classes_, case)
 
    #valid_x[i] = le.transform(valid_x[i])

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
from lightgbm import LGBMRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV

# =============================================================================
# xgb_model = xgboost.XGBRegressor(n_estimators=300, learning_rate=0.01, gamma=0, subsample=0.75,
#                            colsample_bytree=1, max_depth=7) # 587.793413 -> n_estimators=100, learning_rate=0.05, gamma=0, subsample=0.75, colsample_bytree=1, max_depth=7
# 
# print(len(train_x), len(valid_x))
# xgb_model.fit(train_x,train_y)
# 
# xgboost.plot_importance(xgb_model)
# 
# y_pred = xgb_model.predict(test_x)
# 
# r_sq = xgb_model.score(train_x, train_y)
# print(r_sq)
# print(explained_variance_score(predictions,valid_y))
# print(mean_squared_error(valid_y, predictions, squared=False))
# #583.4319932059316 -> n_estimators=200, learning_rate=0.01, gamma=0, subsample=0.75, colsample_bytree=1, max_depth=7
# #583.0324360054929 -> xgboost.XGBRegressor(n_estimators=300, learning_rate=0.01, max_depth=7)
# =============================================================================

lgbm = LGBMRegressor(n_estimators=300, learning_rate=0.01, max_depth=9)

lgbm.fit(train_x, train_y)
y_pred = lgbm.predict(test_x)
print(mean_squared_error(valid_y, y_pred, squared=False))
#580.2095836350069 -> n_estimators=300, learning_rate=0.01, max_depth=9
l_sq = lig_model.score(train_x, train_y)

# SVR 모델 생성 및 훈련
svr = SVR(kernel='rbf', C=250, epsilon=0.2) #가장 rmse 값을 줄이는  파라미터 : kernel='rbf', C=250, epsilon=0.2 -> 598.8620687029053
svr.fit(train_x, train_y)

# 예측 및 성능 평가
y_pred = svr.predict(valid_x)
rmse = mean_squared_error(valid_y, y_pred, squared=False)
print(f'Mean Squared Error: {rmse}')

#param_grid = {'C': [0.1, 1, 10, 100], 'epsilon': [0.1, 0.2, 0.3, 0.4, 0.5], 'kernel': ['linear', 'rbf']}

# GridSearchCV 객체 생성
grid_search = GridSearchCV(SVR(), param_grid, cv=5)

# 그리드 검색으로 모델 훈련
grid_search.fit(train_x, train_y)

# 최적의 하이퍼파라미터 출력
print(f'Best parameters: {grid_search.best_params_}')

#NN 알고리즘 활용
#ensemble=======================================
# 기본 모델 정의
base_models = [
    ('xgb', xgboost.XGBRegressor(n_estimators=300, learning_rate=0.01, gamma=0, subsample=0.75, colsample_bytree=1, max_depth=7)),
    ('lgbm', LGBMRegressor(n_estimators=300, learning_rate=0.01, max_depth=9))
]

# 메타 모델 정의
meta_model = LinearRegression()

# 스태킹 모델 생성
stacking_model = StackingRegressor(estimators=base_models, final_estimator=meta_model)

# 스태킹 모델 훈련
stacking_model.fit(train_x, train_y)

# 예측 및 성능 평가
y_pred = stacking_model.predict(valid_x)
rmse = mean_squared_error(valid_y, y_pred, squared=False)
print(f'Mean Squared Error: {rmse}')
#===============================================
from autosklearn.regression import AutoSklearnRegressor
automl = AutoSklearnRegressor(time_left_for_this_task=120, per_run_time_limit=30)
automl.fit(train_x, train_y)

# 최적의 모델과 하이퍼파라미터 출력
print(automl.show_models())

# 예측 및 성능 평가
y_pred = svr.predict(valid_x)
rmse = mean_squared_error(valid_y, y_pred, squared=False)
print(f'Mean Squared Error: {rmse}')

#====================================

from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingRegressor

# HistGradientBoostingRegressor 모델 생성 및 훈련
hgb = HistGradientBoostingRegressor(
    l2_regularization=3,
    learning_rate=0.015673535665067875,
    max_iter=512,
    max_leaf_nodes=7,
    min_samples_leaf=23,
    n_iter_no_change=0,
    random_state=1,
    validation_fraction=None,
    warm_start=True
)
hgb.fit(train_x, train_y)

# 예측 및 성능 평가
y_pred = hgb.predict(valid_x)
rmse = mean_squared_error(valid_y, y_pred, squared=False)
print(f'Mean Squared Error: {rmse}')

#=============================
# RandomForestRegressor 모델 생성 및 훈련
rf = RandomForestRegressor(
    max_features=0.7517792740390704,
    min_samples_leaf=6,
    n_estimators=512,
    n_jobs=1,
    random_state=1,
    warm_start=True
)
rf.fit(train_x, train_y)

# 예측 및 성능 평가
y_pred = rf.predict(valid_x)
rmse = mean_squared_error(valid_y, y_pred, squared=False)
print(f'Mean Squared Error: {rmse}')

#==================================== # 578.2522232918232 
lgbm = LGBMRegressor(
    learning_rate=0.017,
    num_leaves=7,
    min_child_samples=23,
    n_estimators=300,
)
lgbm = LGBMRegressor(n_estimators=300, learning_rate=0.017, max_depth=9)

lgbm.fit(train_x, train_y)

h_sq2 = lgbm.score(train_x, train_y)
# 예측 및 성능 평가
y_pred = lgbm.predict(valid_x)
rmse = mean_squared_error(valid_y, y_pred, squared=False)
print(f'Mean Squared Error: {rmse}')

y_pred = lgbm.predict(test_x)
#====================================

submission = pd.read_csv('D://Dev//dacon//open//sample_submission.csv')
submission['Income'] = y_pred
submission

submission.to_csv('D://Dev//dacon//open//lgbm_f_submission.csv', index=False)

lf=pd.read_csv('D://Dev//dacon//open//lgbm_f_submission.csv')
lg=pd.read_csv('D://Dev//dacon//open//lgbm_submission.csv')

aaa= pd.concat([lf,lg], axis=1)