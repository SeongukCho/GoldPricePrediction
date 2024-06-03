import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

# CSV파일에서 데이터 불러오기
data_path = 'C:/myAIPrj/GoldPricePrediction/csv/Gold Price (2013-2023).csv'
df = pd.read_csv(data_path)

# 데이터 구조 확인을 위하여 처음 및 행 표시
print(df.head())

# 'Date'열을 datetime 형식으로 변환
df['Date'] = pd.to_datetime(df['Date'])

# 날짜별로 데이터를 정렬하여 시간순으로 정렬
df = df.sort_values('Date')

# 'Date'열을 데이터 프레임의 index로 설정
df.set_index('Date', inplace=True)

# 쉼표를 제거하고 가격을 float 형으로 변환하여 가격 열 정비
df['Price'] = df['Price'].str.replace(',','').astype(float)

# 지정된 기간 동안 금 가격 데이터 구분
plt.figure(figsize=(14,5))
plt.plot(df['Price'])
plt.title('Gold Price (2013-2023)')
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

# 가격 데이터를 신경망 입력에 대해 0과1 범위로 축척
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(df['Price'].values.reshape(-1,1))

# 데이터를 학습용과 검증용으로 분할 (80% 학습, 20% 검증)
train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

# LSTM 모델 학습을 위한 데이터셋 작성
def create_dataset(data, time_step=1):
    x, y = [], []
    for i in range(len(data) - time_step - 1):
        x.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(x), np.array(y)

# LSTM 모델에 대한 시간 단계 수(lookback period) 정의
time_step = 60
x_train, y_train = create_dataset(train_data, time_step)
x_test, y_test = create_dataset(test_data, time_step)

# LSTM에서 요구하는대로 입력 데이터를 [샘플, 시간 단계, 특징] 으로 재구성
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

# 개선된 LSTM 모델 구축
model = Sequential()
# 50개 단위의 첫번째 LSTM 계층, 반환 단계
model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
# 20% 만큼 탈락층을 구성하여 탈락 시킴
model.add(Dropout(0.2))
# 50개 단위의 두번째 LSTM 계층, 반환 단계
model.add(LSTM(50, return_sequences=True))
# 20% 만큼 탈락층을 구성하여 탈락 시킴
model.add(Dropout(0.2))
# 반환 단계가 아닌 50개 단위의 세번째 LSTM 계층
model.add(LSTM(50, return_sequences=False))
# 20% 만큼 탈락층을 구성하여 탈락 시킴
model.add(Dropout(0.2))
# 25개 단위의 조밀한 층
model.add(Dense(25))
# 1단위의 출력 레이어(예상가격)
model.add(Dense(1))

# Adam optimizer와 평균 제곱 오차 손실 함수를 이용하여 모형 컴파일
model.compile(optimizer='adam', loss='mean_squared_error')

# 훈련 데이터로 모델 훈련
model.fit(x_train, y_train, batch_size=32, epochs=50)

# 학습 및 검증데이터 모두에 대한 예측 수행
train_predict = model.predict(x_train)
test_predict = model.predict(x_test)

#  확장된 예측을 원래 가격 수준으로 역변환
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)

# Root Mean Squared Error (RMSE)를 계산하여 모델 성능 평가
train_rmse = np.sqrt(np.mean(((train_predict - scaler.inverse_transform(y_train.reshape(-1, 1))) ** 2)))
test_rmse = np.sqrt(np.mean(((test_predict - scaler.inverse_transform(y_test.reshape(-1, 1))) ** 2)))
print(f'Train RMSE: {train_rmse}')
print(f'Test RMSE: {test_rmse}')

# 그림을 그릴 데이터 준비
train_plot = np.empty_like(scaled_data)
train_plot[:, :] = np.nan
train_plot[time_step:len(train_predict) + time_step, :] = train_predict

test_plot = np.empty_like(scaled_data)
test_plot[:, :] = np.nan
test_plot[len(train_predict) + (time_step * 2) + 1:len(scaled_data) -1, :] = test_predict

# 원본 데이터와 학습 예측 및 검증 예측 그림 그리기
plt.figure(figsize=(14, 5))
plt.plot(scaler.inverse_transform(scaled_data), label='Original Data')
plt.plot(train_plot, label='Train Predict')
plt.plot(test_plot, label='Test Predict')
plt.title('Gold Price Prediction')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()
