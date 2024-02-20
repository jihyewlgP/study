import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import datetime as dt

# 한글 폰트 설정하기
fe = fm.FontEntry(fname = 'MaruBuri-Regular.otf', name = 'MaruBuri')
fm.fontManager.ttflist.insert(0, fe)
plt.rc('font', family='MaruBuri')

online_sales = pd.read_csv('Onlinesales_info.csv')
customers_data = pd.read_csv('Customer_info.csv')
discount_coupon = pd.read_csv('Discount_info.csv')
marketing_spend = pd.read_csv('Marketing_info.csv')
tax_amount = pd.read_csv('Tax_info.csv')

online_sales.head(3)

# 'Transaction_Date' 컬럼을 날짜 타입으로 변환하고 월별로 집계
# 'Month' 컬럼으로 거래량 계산
counts_by_month = online_sales.assign(Month=pd.to_datetime(online_sales['거래날짜']).dt.to_period('M'))['Month'].value_counts().sort_index()

# 시간에 따른 거래량 선 그래프 그리기
plt.figure(figsize=(15, 6))
sns.lineplot(x=counts_by_month.index.astype(str), y=counts_by_month.values)
plt.title('시간에 따른 거래량')
plt.xlabel('월')
plt.ylabel('거래량')
plt.xticks(rotation=45)
plt.show()

gender_counts = customers_data['성별'].value_counts()

# 원그래프 그리기
plt.figure(figsize=(8, 8))  # 그래프 크기 설정
plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', startangle=140)  # 원그래프 그리기
plt.title('성별 시각화')  # 그래프 제목 설정
plt.show()  # 그래프 보여주기

# customers_data에서 '고객지역' 열을 사용하여 지역별 거래량 계산
location_counts = customers_data['고객지역'].value_counts()

# 바 그래프 그리기
plt.figure(figsize=(10, 6))  # 그래프 크기 설정
sns.barplot(x=location_counts.index, y=location_counts.values)  # seaborn을 사용한 바 그래프 그리기
plt.title('고객 지역 시각화')  # 그래프 제목 설정
plt.xlabel('고객지역')  # x축 레이블 설정
plt.xticks(rotation=45)  # x축 레이블 회전
plt.show()  # 그래프 보여주기

# 원본 데이터셋 복사
rfm_online_sales = online_sales.copy()

# 날짜 형식 변환
rfm_online_sales['거래날짜'] = pd.to_datetime(rfm_online_sales['거래날짜'])

# 데이터 내 마지막 날짜 계산
last_date = rfm_online_sales['거래날짜'].max()

# Recency 계산
recency_data = rfm_online_sales.groupby('고객ID')['거래날짜'].max().reset_index()
recency_data['Recency'] = (last_date - recency_data['거래날짜']).dt.days

# Frequency 계산
frequency_data = rfm_online_sales.groupby('고객ID')['거래ID'].count().reset_index()
frequency_data.rename(columns={'거래ID': 'Frequency'}, inplace=True)

# Monetary 계산
rfm_online_sales['SalesValue'] = rfm_online_sales['수량'] * rfm_online_sales['평균금액']
monetary_data = rfm_online_sales.groupby('고객ID')['SalesValue'].sum().reset_index()
monetary_data.rename(columns={'SalesValue': 'Monetary'}, inplace=True)

# RFM 데이터 결합
rfm_data = recency_data.merge(frequency_data, on='고객ID').merge(monetary_data, on='고객ID')

# Recency, Frequency, Monetary에 점수 부여
rfm_data['R_Score'] = pd.qcut(rfm_data['Recency'], 4, labels=[4, 3, 2, 1])
rfm_data['F_Score'] = pd.qcut(rfm_data['Frequency'], 4, labels=[1, 2, 3, 4])
rfm_data['M_Score'] = pd.qcut(rfm_data['Monetary'], 4, labels=[1, 2, 3, 4])

# RFM 스코어 계산
rfm_data['RFM_Score'] = rfm_data['R_Score'].astype(str) + rfm_data['F_Score'].astype(str) + rfm_data['M_Score'].astype(str)

# 고객 세그먼트 분류 함수
def classify_customer_segment(row):
    R, F, M = row['R_Score'], row['F_Score'], row['M_Score']
    
    if R == 4 and F == 4 and M == 4:
        return 'VIP고객'
    elif R >= 2 and F >= 3 and M == 4:
        return '충성고객'
    elif R >= 3 and F <= 3 and M <= 3:
        return '잠재충성고객'
    elif R == 2 and F < 2 and M < 2:
        return '신규고객'
    elif R >= 3 and F < 2 and M < 2:
        return '정체고객'
    elif R >= 3 and F >= 3 and M >= 3:
        return '관심필요고객'
    elif 2 <= R <= 3 and F < 3 and M < 3:
        return '잠드려는고객'
    elif R < 3 and 2 <= F <= 4 and 2 <= M <= 4:
        return '이탈우려고객'
    elif R < 2 and F == 4 and M == 4:
        return '놓치면안될고객'
    elif 2 <= R <= 3 and 2 <= F <= 3 and 2 <= M <= 3:
        return '겨울잠고객'
    elif R < 2 and F < 2 and M < 2:
        return '이탈고객'
    else:
        return '기타'
    
# rfm_data에 'Customer_Segment' 컬럼 추가
rfm_data['Customer_Segment'] = rfm_data.apply(classify_customer_segment, axis=1)

# 가능한 모든 세그먼트 정의
all_segments = ['VIP고객', '충성고객', '잠재충성고객', '신규고객', '정체고객', '관심필요고객', 
                '잠드려는고객', '이탈우려고객', '놓치면안될고객', '겨울잠고객', '이탈고객', '기타']

# 각 세그먼트별 고객 수 계산
segment_counts = rfm_data['Customer_Segment'].value_counts()

# 모든 세그먼트에 대한 고객 수를 0으로 초기화하고, 계산된 값으로 업데이트
segment_counts_all = {segment: 0 for segment in all_segments}
segment_counts_all.update(segment_counts)

# 데이터프레임으로 변환
segment_counts_df = pd.DataFrame(list(segment_counts_all.items()), columns=['Segment', 'Count'])

# 세그먼트별 고객 수를 내림차순으로 정렬
segment_counts_sorted_df = segment_counts_df.sort_values(by='Count', ascending=False)

# 바 그래프 시각화
plt.figure(figsize=(15, 8))
plt.bar(segment_counts_sorted_df['Segment'], segment_counts_sorted_df['Count'], color='skyblue')
plt.title('고객 세그먼트별 분포')
plt.xlabel('고객 세그먼트')
plt.ylabel('고객 수')
plt.xticks(rotation=45)
plt.show()

##########################################################
import pandas as pd
from datetime import timedelta

# 데이터 로드 (고객 ID, 구매 날짜, 구매 금액) 고객의 구매 패턴을 분석 방법
df = pd.read_csv('purchase_data.csv')

# 현재 날짜 설정 (가장 최근의 구매일 + 1일)
snapshot_date = df['PurchaseDate'].max() + timedelta(days=1)

# RFM 계산
rfm = df.groupby(['CustomerID']).agg({
    'PurchaseDate': lambda x: (snapshot_date - x.max()).days,
    'InvoiceNo': 'count',
    'TotalPurchase': 'sum'})

# 컬럼명 변경
# Recency(최근성), Frequency(구매횟수), Monetary(구매금액)
rfm.rename(columns={'PurchaseDate': 'Recency',
                    'InvoiceNo': 'Frequency',
                    'TotalPurchase': 'MonetaryValue'}, inplace=True)

# cohort 분석 특정 기간 동안 같은 경험을 한 사용자 그룹을 추적하는 분석 방법
# 주문 월과 Cohort 그룹 정의
def get_month(x): 
    return dt.datetime(x.year, x.month, 1)
df['OrderMonth'] = df['PurchaseDate'].apply(get_month)
df['CohortMonth'] = df.groupby('CustomerID')['OrderMonth'].transform('min')

# Cohort Index 계산
def get_date_int(df, column):
    year = df[column].dt.year
    month = df[column].dt.month
    return year, month

order_year, order_month = get_date_int(df, 'OrderMonth')
cohort_year, cohort_month = get_date_int(df, 'CohortMonth')
years_diff = order_year - cohort_year
months_diff = order_month - cohort_month
df['CohortIndex'] = years_diff * 12 + months_diff + 1

# Cohort 테이블 생성
cohort_data = df.groupby(['CohortMonth', 'CohortIndex'])['CustomerID'].apply(pd.Series.nunique).reset_index()
cohort_count = cohort_data.pivot_table(index='CohortMonth', columns='CohortIndex', values='CustomerID')

#의미 있는 인사이트를 도출하고 그 결과를 바탕으로 혁신적인 비즈니스 솔루션 제안