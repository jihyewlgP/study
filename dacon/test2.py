# -*- coding: utf-8 -*- 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import sklearn.metrics as mt 
from sklearn.tree import export_graphviz 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import StratifiedShuffleSplit 
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV 
#from sklearn.externals import joblib 
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import mean_squared_error
import seaborn as sns

import pandas as pd

train = pd.read_csv('D://Dev//dacon//open//train.csv')
test = pd.read_csv('D://Dev//dacon//open//test.csv')

X_train = train.drop(columns=['ID', 'Income'])
X_valid = train.drop(columns=['ID', 'Income'])
y_train = train['Income']

plt.figure(figsize=(16, 12))
sns.heatmap(train.corr(), annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

X_train = X_train.drop(columns=['Household_Summary', 'Household_Status', 'Income_Status', 'Citizenship', 'Hispanic_Origin', 'Race'])
X_valid = X_valid.drop(columns=['Household_Summary', 'Household_Status', 'Income_Status', 'Citizenship', 'Hispanic_Origin', 'Race'])

X_train = X_train[:14000]
y_train = train['Income'][:14000]

X_valid = X_valid[14000:]
y_valid = train['Income'][14000:]

X_test = test.drop(columns=['ID'])
X_test = X_test.drop(columns=['Household_Summary', 'Household_Status', 'Income_Status', 'Citizenship', 'Hispanic_Origin', 'Race'])

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
    
# =============================================================================
    for case in np.unique(X_valid[i]):
        if case not in le.classes_: 
            le.classes_ = np.append(le.classes_, case)
 
    X_valid[i] = le.transform(X_valid[i])
# =============================================================================

    # test 데이터의 새로운 카테고리에 대해 le.classes_ 배열에 추가
    for case in np.unique(X_test[i]):
        if case not in le.classes_: 
            le.classes_ = np.append(le.classes_, case)
    
    X_test[i] = le.transform(X_test[i])

# ===== 랜덤포레스트 메인 ===== 
# 4. 모델 세부 튜닝: 최적 하이퍼파라미터 찾기
rnd_clf = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42) 
 
param_dist_rf = { 
 'n_estimators':[350, 400, 450, 500], 
 'max_leaf_nodes':[60, 70, 80, 90, 120, 150], 
 'max_features':[6, 7, 8, 9] 
}
# 
rnd_search = RandomizedSearchCV(rnd_clf, param_dist_rf, cv=10, random_state=42) 
rnd_search.fit(X_train, y_train) 
print(rnd_search.best_params_)
# 300 50 10 - 570.4261696313174
# 300 60 11 - 569.9428554305892
# 400 70 8 - 569.8671296994535
# 400 80 8 - 569.6122831226205
# 350 65 11 - 569.9211682108541
# 400 100 8 - 569.21504814323
# 400 120 8 - 569.1515687400126
# 400 120 9 - 569.0449277150102 너무 높음 성능 낮음
# 400 150 8 - 569.3064305074203

# 5. 학습 및 K-fold cross_validation 평가 
rnd_clf = RandomForestRegressor(n_estimators=400, max_leaf_nodes=150, max_features=8, n_jobs=-1, random_state=42) #디폴트
rnd_scores = cross_val_score(rnd_clf, X_train, y_train, scoring="accuracy", cv=10)
print("\n<10-fold cross-validation>") 
print("accuracy score mean: ", rnd_scores.mean()) 
# 6. 최종 모델 학습
rnd_clf.fit(X_train, y_train) 
print("\n<AI model: machine learning done >") 
print("accuracy_score of train data(0.8 of sample): ", rnd_clf.score(X_train, y_train)) 
# 7. test data 확인
print("accuracy_score of test data(0.2 of sample): ", rnd_clf.score(X_valid, y_valid)) 
#y_test_pred = rnd_clf.predict(X_test) 
#print("accuracy_score of test data: ", mt.accuracy_score(y_test, y_test_pred)) 
# 8. confusion matrix 확인
y_test_pred = rnd_clf.predict(X_valid) 

rmse = mean_squared_error(y_valid, y_test_pred, squared=False)
print(f'Root Mean Squared Error: {rmse}')

# =============================================================================
# cm1= confusion_matrix(y_valid, y_test_pred, labels=["up","neutral","down"]) 
# print("\n<Confusion matrix>") 
# print("(of test)") 
# print("up","neutral","down") 
# print(cm1) 
# cm2= confusion_matrix(y_past, rnd_clf.predict(X_past), labels=["up","neutral","down"]) 
# print("(of all)") 
# print("up","neutral","down") 
# print(cm2) 
# # 9. 변수 중요도 체크
# print("\n<Feature importance>") 
# for name, score in zip(X.columns, rnd_clf.feature_importances_): 
#  print(name, ": ", score) 
# # 10. backtesting용 과거의 예측데이터 생성
# 
# y_prediction = rnd_clf.predict(X) 
# y_pred = pd.Series(y_prediction, index=y.index) 
# # 11. 모델 저장
# joblib.dump(rnd_clf, "forecast_model.pkl") 
# print("\n< AI model: save >") 
# =============================================================================

submission = pd.read_csv('D://Dev//dacon//open//sample_submission.csv')
submission['Income'] = preds
submission

submission.to_csv('D://Dev//dacon//open//preprocess_submission.csv', index=False)

lf=pd.read_csv('D://Dev//dacon//open//lgbm_f_submission.csv')
lg=pd.read_csv('D://Dev//dacon//open//lgbm_submission.csv')

#===========================================
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
#======================
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

#==================================
# 데이터 피처링 완료 후 모델링
from lightgbm import LGBMRegressor, early_stopping

X_train=X_train.drop(columns=['Gains', 'Dividends'])
X_valid=X_valid.drop(columns=['Gains', 'Dividends'])
# LightGBM 모델 초기화
model = LGBMRegressor()

# 튜닝할 하이퍼파라미터 설정
param_grid = {
    'n_estimators': [100, 200, 300, 408],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [5, 7, 9]
}

# GridSearchCV를 사용하여 최적의 하이퍼파라미터 찾기
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', verbose=2)
grid_search.fit(X_train, y_train)

# 최적의 하이퍼파라미터로 모델 학습
model = grid_search.best_estimator_

model = LGBMRegressor(n_estimators=400, learning_rate=0.01, max_depth=10)

# 모델 학습 (조기 종료 사용)
model.fit(X_train, y_train)#, eval_set=[(X_valid, y_valid)], eval_metric='rmse', callbacks=[early_stopping(stopping_rounds=50)])

# 테스트 데이터에 대한 예측 생성
preds = model.predict(X_test)

# RMSE 계산
rmse = np.sqrt(mean_squared_error(y_valid, preds))
print("RMSE: %f" % (rmse))