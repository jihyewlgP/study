### Dacon - 소득예측 데이터 분석 진행

소득예측을 위해 y_value = Income 지정
X_value 값을 비교하기 위해
```
train.columns
```
['ID', 'Age', 'Gender', 'Education_Status', 'Employment_Status',
       'Working_Week (Yearly)', 'Industry_Status', 'Occupation_Status', 'Race',
       'Hispanic_Origin', 'Martial_Status', 'Household_Status',
       'Household_Summary', 'Citizenship', 'Birth_Country',
       'Birth_Country (Father)', 'Birth_Country (Mother)', 'Tax_Status',
       'Gains', 'Losses', 'Dividends', 'Income_Status', 'Income']

***
['ID', '나이', '성별', '교육_상태', '고용_상태',
'근무시간_주간(연간)', '산업_현황', '직업_현황', '인종',
'Hispanic_Origin', '결혼상태', 'Household_Status',
'가계_요약', '시민권', '출생_나라',
'출생_국가(아버지)', '출생_국가(어머니)', '세금_상태',
'이득', '손실', '배당', '소득_상태', '소득']

### 가설 설정을 위해 EDA 진행

1. info를 통해 확인한 object 컬럼 변수들의 unique 정보 확인
```
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

train.info() # 결측치 없음
test.info() # 결측치 household_status에 존재함
```
* 타겟 변수인 'income' 분포 시각화
```
# 타겟 변수인 'y'의 분포를 시각화
plt.figure(figsize=(10, 6))
sns.histplot(train['Income'], bins=30, kde=True)
plt.title('Distribution of y')
plt.show()
```

```
print('Gender \n',train['Gender'].unique())

print('Education_Status \n',train['Education_Status'].unique())

print('Employment_Status \n',train['Employment_Status'].unique())

print('Industry_Status \n',train['Industry_Status'].unique()) # vif 높음

print('Occupation_Status \n',train['Occupation_Status'].unique())

print('Race \n',train['Race'].unique())

print('Hispanic_Origin \n',train['Hispanic_Origin'].unique())

print('Martial_Status \n',train['Martial_Status'].unique())

print('Household_Status \n',train['Household_Status'].unique()) # 중복 summary 와 같은 내용임

print('Household_Summary \n',train['Household_Summary'].unique())

print('Citizenship \n',train['Citizenship'].unique())

print('Birth_Country \n',train['Birth_Country'].unique()) #vif 높음

print('Birth_Country (Father) \n',train['Birth_Country (Father)'].unique())

print('Birth_Country (Mother) \n',train['Birth_Country (Mother)'].unique())

print('Tax_Status \n',train['Tax_Status'].unique()) #vif 높음

print('Income_Status \n',train['Income_Status'].unique())
```
2. 각 변수와 income 소득간의 분포를 확인
```
plt.figure(figsize=(12, 10))
plt.title('Education_Status_and_Income', fontsize = 30)
sns.scatterplot(x = 'Education_Status',y= 'Income', data= train)
```
3. 태어난 나라의 분포를 확인
```
native_country_table=train['Birth_Country'].value_counts()
native_country_table

colors = sns.color_palette('pastel')[0:42]
#pie chart 확인결과도 동일함을 확인
plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.pie(train['Birth_Country'].value_counts(),labels=native_country_table.index,colors=colors)
plt.subplot(1,2,2)
sns.countplot(data=train, x="Birth_Country")
```
* 결과 불균형하게 분포하고 있음을 확인
즉, US, Mexico에 주로 분포하고 있음을 확인

4. 상관관계 분석을 위해 X_value값의 데이터를 label_encoder를 진행

```
# X_valid는 실제 결과에는 들어가지 않는 정확도를 확인하기 위함
# 따라서 실제 test 값을 위해서는 제외 후 진행

X_train = X_train[:14000]
y_train = train['Income'][:14000]

X_valid = X_valid[14000:]
y_valid = train['Income'][14000:]

from sklearn.preprocessing import LabelEncoder

encoding_target = list(X_train.dtypes[X_train.dtypes == "object"].index)

for i in encoding_target:
    le = LabelEncoder()
    
    # train과 test 데이터셋에서 해당 열의 모든 값을 문자열로 변환
    X_train[i] = X_train[i].astype(str)
    X_valid[i] = X_valid[i].astype(str)
    X_test[i] = X_test[i].astype(str)
    
    le.fit(X_train[i])
    X_train[i] = le.transform(X_train[i])

    for case in np.unique(X_valid[i]):
        if case not in le.classes_: 
            le.classes_ = np.append(le.classes_, case)
 
    X_valid[i] = le.transform(X_valid[i])

    # test 데이터의 새로운 카테고리에 대해 le.classes_ 배열에 추가
    for case in np.unique(X_test[i]):
        if case not in le.classes_: 
            le.classes_ = np.append(le.classes_, case)
    
    X_test[i] = le.transform(X_test[i])
```
* 상관관계분석 시각화
```
plt.figure(figsize=(16, 12))
sns.heatmap(train.corr(), annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()
```
* 상관관계가 높은 값을 주요변수로 선택 후 진행 필요
각 변수가 소득에 큰 영향을 주지 않을 경우 drop 진행
중요변수로 선택시 vif 다중공선성을 확인할 필요가 있음.

#### * 회귀분석을 통한 vif 도출
```
#회귀분석
from statsmodels.formula.api import ols

#각 변수를 전진선택법을 통해 성능 지표를 확인하며 vif가 10이상인 변수 제거
model = ols("Income ~ {}+ {} + {} + {} + {} + {} + {} + {}+{}+{}+{}+{}".format("Martial_Status","Education_Status",'Occupation_Status', 'Race', 'Hispanic_Origin', 'Martial_Status', 'Citizenship','Household_Summary','Income_Status','Employment_Status', 'Gender', 'Age'), train)
res = model.fit()
res.summary()
#Birth_Country 다중공선성을 넘김.

from statsmodels.stats.outliers_influence import variance_inflation_factor
aa=pd.DataFrame({'컬럼': column, 'VIF': variance_inflation_factor(model.exog, i)} 
             for i, column in enumerate(model.exog_names)
             if column != 'Intercept')  
print(aa[aa['VIF']>10])
```
***
각 다중공선성이 높은 컬럼은 
['Household_Status', 'Gains', 'Losses', 'Dividends', 'Birth_Country', 'Birth_Country (Father)', 'Birth_Country (Mother)', 'Industry_Status', 'Tax_Status']

학습시 필요없는 컬럼
['ID', 'Income']

상관관계가 가장 높은 컬럼
['Working_Week (Yearly)']

* 주 근무시간이 길수록 소득이 높은것을 확인 가능
* 소득 예측에 중요한 유의미성을 지닌 변수
***
* 이상치 확인 및 그래프 확인
```
import matplotlib.pyplot as plt

df = pd.concat([X_train,y_train], axis=1)

# 박스 플롯으로 이상치 확인
plt.figure(figsize=(10, 4))
sns.boxplot(x=df["Age"])
plt.show()

# IQR 방법으로 이상치 확인
Q1 = df['Age'].quantile(0.25)
Q3 = df['Age'].quantile(0.75)
IQR = Q3 - Q1
outliers = df[(df['Age'] < (Q1 - 1.5 * IQR)) | (df['Age'] > (Q3 + 1.5 * IQR))]
print(outliers)

# 남성과 여성의 평균 소득 계산
average_income = df.groupby('Gender')['Income'].mean()

# 바 차트 생성
average_income.plot(kind='bar', color=['blue', 'pink'])

# 차트 제목과 레이블 설정
plt.title('Average Income by Gender')
plt.xlabel('Gender')
plt.ylabel('Average Income')

# 차트 보여주기
plt.show()

# 박스 플롯으로 이상치 확인
plt.figure(figsize=(10, 4))
sns.boxplot(x=df["Education_Status"])
plt.show()

# IQR 방법으로 이상치 확인
Q1 = df['Education_Status'].quantile(0.25)
Q3 = df['Education_Status'].quantile(0.75)
IQR = Q3 - Q1
outliers = df[(df['Education_Status'] < (Q1 - 1.5 * IQR)) | (df['Education_Status'] > (Q3 + 1.5 * IQR))]
print(outliers)

# 박스 플롯으로 이상치 확인
plt.figure(figsize=(10, 4))
sns.boxplot(x=df["Employment_Status"])
plt.show()

# IQR 방법으로 이상치 확인
Q1 = df['Employment_Status'].quantile(0.25)
Q3 = df['Employment_Status'].quantile(0.75)
IQR = Q3 - Q1
outliers = df[(df['Employment_Status'] < (Q1 - 1.5 * IQR)) | (df['Employment_Status'] > (Q3 + 1.5 * IQR))]
print(outliers)

# y와 상관관계가 높은 피처만 선택
correlation = df.corr()['Income'].abs().sort_values(ascending=False)
features = correlation[correlation > 0.1].index

# 다중공선성 계산
from statsmodels.stats.outliers_influence import variance_inflation_factor

# VIF를 계산할 피처 선택
features = df[['Income', 'Working_Week (Yearly)', 'Occupation_Status',
       'Martial_Status', 'Tax_Status', 'Household_Status', 'Age',
       'Household_Summary']]

# VIF 계산
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(features.values, i) for i in range(features.shape[1])]
vif["features"] = features.columns

# 다중공선성이 높은 Household_Summary, Household_Status 는 제거한다.
df = df.drop(columns=['Household_Summary', 'Household_Status'])

# 박스 플롯으로 이상치 확인
plt.figure(figsize=(10, 4))
sns.boxplot(x=df["Birth_Country (Mother)"])
plt.show()

plt.figure(figsize=(10, 4))
sns.boxplot(x=df["Birth_Country"])
plt.show()

# IQR 방법으로 이상치 확인
Q1 = df['Employment_Status'].quantile(0.25)
Q3 = df['Employment_Status'].quantile(0.75)
IQR = Q3 - Q1
outliers = df[(df['Employment_Status'] < (Q1 - 1.5 * IQR)) | (df['Employment_Status'] > (Q3 + 1.5 * IQR))]
print(outliers)

# 태어난 나라에 따른 평균 소득 계산
average_income = df.groupby('Employment_Status')['Income'].mean()

# 바 차트 생성
average_income.plot(kind='bar')

# 차트 제목과 레이블 설정
plt.title('Average Income by Employment_Status')
plt.xlabel('Employment_Status')
plt.ylabel('Average Income')

# 의미 없는 데이터들 삭제
df = df.drop(columns=['Income_Status', 'Citizenship', 'Hispanic_Origin', 'Race'])
```
***
#### 변수제거 없이 회귀분석 진행
```
from sklearn.metrics import mean_squared_error

import pandas as pd

train = pd.read_csv('D://Dev//dacon//open//train.csv')
test = pd.read_csv('D://Dev//dacon//open//test.csv')

train_x = train.drop(columns=['ID', 'Income'])
valid_x = train.drop(columns=['ID', 'Income']) # 학습 예측 결과 정확도 확인
train_y = train['Income']

train_x = train_x[:14000]
train_y = train['Income'][:14000]

valid_x = valid_x[14000:]
valid_y = train['Income'][14000:]

test_x = test.drop(columns=['ID'])

from sklearn.preprocessing import LabelEncoder

encoding_target = list(train_x.dtypes[train_x.dtypes == "object"].index)

for i in encoding_target:
    le = LabelEncoder()
    
    # train과 test 데이터셋에서 해당 열의 모든 값을 문자열로 변환
    train_x[i] = train_x[i].astype(str)
    valid_x[i] = valid_x[i].astype(str)
    test_x[i] = test_x[i].astype(str)
    
    le.fit(train_x[i])
    train_x[i] = le.transform(train_x[i])
    
# =============================================================================
    for case in np.unique(valid_x[i]):
        if case not in le.classes_: 
            le.classes_ = np.append(le.classes_, case)
 
    valid_x[i] = le.transform(valid_x[i])
# =============================================================================

    # test 데이터의 새로운 카테고리에 대해 le.classes_ 배열에 추가
    for case in np.unique(test_x[i]):
        if case not in le.classes_: 
            le.classes_ = np.append(le.classes_, case)
    
    test_x[i] = le.transform(test_x[i])
    
lgbm = LGBMRegressor(random_state=1, warm_start=True, n_estimators=512, learning_rate=0.01, max_depth=9)

lgbm.fit(train_x, train_y)
score = lgbm.score(train_x, train_y)

# RMSE 정확도가 낮을 수록 성능이 높음

lgbm = LGBMRegressor(n_estimators=300, learning_rate=0.01, max_depth=9)
# RMSE : 562.917517226174

lgbm = LGBMRegressor(n_estimators = 440(408) learning_rate = 0.01, max_depth=9)
# RMSE : 561.6376010000113
=> 등수 16등 -> 65등

* 해당 561.92 성능보다 높이기 위해서 RMSE 값 최적화 필요
* 하이퍼파라미터 수정, feature tuning 필요
* 또는 ensemble 진행 시 최적화 가능할 수 있음
```

#### Ensemble 시 성능 최적화 가능?
```
#Library
from catboost import CatBoostRegressor
from sklearn.ensemble import BaggingRegressor
from category_encoders import CatBoostEncoder
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import VotingRegressor
from sklearn.metrics import mean_squared_error

cat_features = ['Gender','Education_Status','Employment_Status','Industry_Status','Occupation_Status', 'Race', 'Hispanic_Origin', 'Martial_Status', 'Household_Summary', 'Citizenship', 'Birth_Country', 'Birth_Country (Father)', 'Birth_Country (Mother)', 'Tax_Status', 'Income_Status']

model = CatBoostRegressor(iterations=300, depth=7, learning_rate=0.05, loss_function='RMSE', random_state=0)
bagging_model = BaggingRegressor(base_estimator=model, n_estimators=10, random_state=0)
lgb_model = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.01, max_depth=7, random_state=0)
xgb_model = xgb.XGBRegressor(n_estimators=300, learning_rate=0.01, max_depth=7, random_state=0)

ensemble_models = [
    ('catboost', model),
    ('bagging', bagging_model),
    ('lgbm', lgb_model),
    ('xgboost', xgb_model)
    ]

#bagging_model.fit(X_train, y_train)#, cat_features)
voting_model = VotingRegressor(estimators=ensemble_models)
voting_model.fit(X_train, y_train)

preds = voting_model.predict(X_valid)

rmse = mean_squared_error(y_valid, preds, squared=False)
print(f'Root Mean Squared Error: {rmse}')
```
* CatBoostRegressor 값 
 > #563.2499298307571 - 400 9 0.05
  #565.0895185196862 - 400 11 0.05
  #563.859307763338 - 400 10 0.05
  #566.0607345558554 - 400 9 0.1
  #562.8880782880971 - 500 9 0.05
  #562.818021244306 - 600 9 0.05

* 기존 LightGbm으로만 예측한 값보다 낮은 성능을 가짐

#### RandomForestRegressor
```
rnd_clf = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42) 
 
param_dist_rf = { 
 'n_estimators':[350, 400, 450, 500], 
 'max_leaf_nodes':[60, 70, 80, 90, 120, 150], 
 'max_features':[6, 7, 8, 9] 
}

rnd_search = RandomizedSearchCV(rnd_clf, param_dist_rf, cv=10, random_state=42) 
rnd_search.fit(X_train, y_train) 
print(rnd_search.best_params_)

rnd_clf.fit(X_train, y_train) 

y_test_pred = rnd_clf.predict(X_valid) 

rmse = mean_squared_error(y_valid, y_test_pred, squared=False)
print(f'Root Mean Squared Error: {rmse}')
```
>#300 50 10 - 570.4261696313174
#300 60 11 - 569.9428554305892
#400 70 8 - 569.8671296994535
#400 80 8 - 569.6122831226205
#350 65 11 - 569.9211682108541
#400 100 8 - 569.21504814323
#400 120 8 - 569.1515687400126
#400 120 9 - 569.0449277150102
#400 150 8 - 569.3064305074203
* 모두 성능이 최적화 되지 않음

#### XGBRegressor
```
xgb_model = xgboost.XGBRegressor(n_estimators=300, learning_rate=0.01, gamma=0, subsample=0.75,colsample_bytree=1, max_depth=7) 
# 587.793413 -> n_estimators=100, learning_rate=0.05, gamma=0, subsample=0.75, colsample_bytree=1, max_depth=7
 
print(len(train_x), len(valid_x))
xgb_model.fit(train_x,train_y)

xgboost.plot_importance(xgb_model)

y_pred = xgb_model.predict(test_x)

r_sq = xgb_model.score(train_x, train_y)
print(r_sq)
print(explained_variance_score(predictions,valid_y))
print(mean_squared_error(valid_y, predictions, squared=False))
#583.4319932059316 -> n_estimators=200, learning_rate=0.01, gamma=0, subsample=0.75, colsample_bytree=1, max_depth=7
#583.0324360054929 -> xgboost.XGBRegressor(n_estimators=300,learning_rate=0.01, max_depth=7)
```
```
import xgboost as xgb

# XGBoost 회귀 모델을 초기화
xg_reg = xgb.XGBRegressor()


# 튜닝할 하이퍼파라미터 설정
param_grid = {
    'n_estimators': [100, 150, 200],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [5, 7, 9],
    'colsample_bytree': [0.3, 0.5, 0.6, 0.7],
    'gamma': [0.1, 0.2, 0.05]
}

# GridSearchCV를 사용하여 최적의 하이퍼파라미터 찾기
grid_search = GridSearchCV(estimator=xg_reg, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', verbose=2)
grid_search.fit(X_train, y_train)

# 최적의 하이퍼파라미터로 모델 학습
xg_reg = grid_search.best_estimator_

xg_reg = xgb.XGBRegressor(colsample_bytree= 0.5,gamma= 0.1,learning_rate= 0.05, max_depth= 7, n_estimators= 500)
xg_reg.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], eval_metric='rmse', early_stopping_rounds=10)

# 테스트 데이터에 대한 예측 생성
preds = xg_reg.predict(X_valid)

# RMSE 계산
rmse = np.sqrt(mean_squared_error(y_valid, preds))
print("RMSE: %f" % (rmse))

#{'colsample_bytree': 0.5,'gamma': 0.1,'learning_rate': 0.05,'max_depth': 7,'n_estimators': 100} -> 563.391252
```

***

#### 테스트 결과 저장
* RMSE 값이 가장 낮은 모델로 예측
* valid 분리 시키지 않은 데이터를 재학습 후 결과 저장 후 제출
```
y_pred = lgbm.predict(test_x)

submission = pd.read_csv('D://Dev//dacon//open//sample_submission.csv')
submission['Income'] = y_pred
submission
 submission.to_csv('D://Dev//dacon//open//two_submission.csv', index=False)
```