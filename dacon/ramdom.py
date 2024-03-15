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
from sklearn.model_selection import RandomizedSearchCV 
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

X_train = X_train[:14000]
y_train = train['Income'][:14000]

X_valid = X_valid[14000:]
y_valid = train['Income'][14000:]

X_test = test.drop(columns=['ID'])

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
 'n_estimators':[300, 350, 400, 450], 
 'max_leaf_nodes':[40, 50, 60, 70, 80, 90], 
 'max_features':[6, 7, 8, 9, 10, 11] 
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
# 400 120 9 - 569.0449277150102

# 5. 학습 및 K-fold cross_validation 평가 
rnd_clf = RandomForestRegressor(n_estimators=400, max_leaf_nodes=120, max_features=9, n_jobs=-1, random_state=42) #디폴트
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
y_test_pred = rnd_clf.predict(X_test) 

rmse = mean_squared_error(y_valid, y_test_pred, squared=False)
print(f'Root Mean Squared Error: {rmse}')

#=====================

cm1= confusion_matrix(y_valid, y_test_pred, labels=["up","neutral","down"]) 
print("\n<Confusion matrix>") 
print("(of test)") 
print("up","neutral","down") 
print(cm1) 
cm2= confusion_matrix(y_past, rnd_clf.predict(X_past), labels=["up","neutral","down"]) 
print("(of all)") 
print("up","neutral","down") 
print(cm2) 
# 9. 변수 중요도 체크
print("\n<Feature importance>") 
for name, score in zip(X.columns, rnd_clf.feature_importances_): 
 print(name, ": ", score) 
# 10. backtesting용 과거의 예측데이터 생성

y_prediction = rnd_clf.predict(X) 
y_pred = pd.Series(y_prediction, index=y.index) 
# 11. 모델 저장
joblib.dump(rnd_clf, "forecast_model.pkl") 
print("\n< AI model: save >") 


submission = pd.read_csv('D://Dev//dacon//open//sample_submission.csv')
submission['Income'] = y_test_pred
submission

submission.to_csv('D://Dev//dacon//open//rdf_submission.csv', index=False)

lf=pd.read_csv('D://Dev//dacon//open//lgbm_f_submission.csv')
lg=pd.read_csv('D://Dev//dacon//open//lgbm_submission.csv')