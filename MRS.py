import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Dot, Dense
from tensorflow.keras.optimizers import Adam

# 데이터 불러오기
movies = pd.read_csv('csv/tmdb_5000_movies.csv')
credits = pd.read_csv('csv/tmdb_5000_credits.csv')

# 필요한 컬럼만 선택
movies = movies[['id', 'title', 'vote_average', 'vote_count', 'popularity']]
credits = credits[['movie_id']]

# 'id'를 기준으로 병합
data = pd.merge(movies, credits, left_on='id', right_on='movie_id')

# vote_count가 1000 이상인 영화만 선택
m = 1000
data = data[data['vote_count'] >= m]

# 전체 영화의 평균 평점 계산
C = data['vote_average'].mean()

# 가중 평점 계산 함수 정의
def weighted_rating(x, m=m, C=C):
    v = x['vote_count']
    R = x['vote_average']
    return (v / (v + m) * R) + (m / (v + m) * C)

# 데이터프레임에 가중 평점 추가
data['score'] = data.apply(weighted_rating, axis=1)

# 가중 평점에 따라 정렬
data = data.sort_values('score', ascending=False)

# 상위 10개 영화 추천 (score 기준)
top_10_movies_score = data[['title', 'score', 'popularity']].head(10)

# popularity에 따라 정렬
data = data.sort_values('popularity', ascending=False)

# 상위 10개 영화 추천 (popularity 기준)
top_10_movies_popularity = data[['title', 'score', 'popularity']].head(10)

print("Top 10 Movies by Weighted Score")
print(top_10_movies_score)

print("Top 10 Movies by Popularity")
print(top_10_movies_popularity)

# 예제용 영화 평점 데이터 생성 (score 사용)
data['userId'] = np.random.randint(0, 100, data.shape[0])
data['rating'] = data['score']

# train, test 데이터셋 분리
train, test = train_test_split(data, test_size=0.2, random_state=42)

# 고유한 사용자와 영화 개수 확인
max_user_id = data['userId'].max() + 1
max_movie_id = data['id'].max() + 1

# 모델 구성
user_input = Input(shape=(1,), name='user_input')
movie_input = Input(shape=(1,), name='movie_input')

user_embedding = Embedding(input_dim=max_user_id, output_dim=50, name='user_embedding')(user_input)
movie_embedding = Embedding(input_dim=max_movie_id, output_dim=50, name='movie_embedding')(movie_input)

user_vec = Flatten(name='user_vec')(user_embedding)
movie_vec = Flatten(name='movie_vec')(movie_embedding)

dot_user_movie = Dot(axes=1, name='dot_user_movie')([user_vec, movie_vec])

output = Dense(1, activation='linear', name='output')(dot_user_movie)

model = Model(inputs=[user_input, movie_input], outputs=output)
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# 모델 학습
histroy = model.fit([train['userId'], train['id']], train['rating'],
                    batch_size=64,
                    epochs=10,
                    verbose=1,
                    validation_split=0.2)

# 그래프 그리기 (score 기준)
plt.figure(figsize=(14, 8))
ax1 = top_10_movies_score['score'].plot(kind='bar', color='skyblue')
ax1.set_title('Top 10 Movies by Weighted Score')
ax1.set_xlabel('Movie Title')
ax1.set_ylabel('Weighted Score')
ax1.set_xticklabels(top_10_movies_score['title'], rotation=45)
for i, score in enumerate(top_10_movies_score['score']):
    ax1.text(i, score + 0.01, f'{score:.2f}', ha='center', va='bottom', fontsize=10)
plt.show()

# 그래프 그리기 (popularity 기준)
plt.figure(figsize=(14, 8))
ax2 = top_10_movies_popularity['popularity'].plot(kind='bar', color='lightgreen')
ax2.set_title('Top 10 Movies by Popularity')
ax2.set_xlabel('Movie Title')
ax2.set_ylabel('Popularity')
ax2.set_xticklabels(top_10_movies_popularity['title'], rotation=45)
for i, popularity in enumerate(top_10_movies_popularity['popularity']):
    ax2.text(i, popularity + 1, f'{popularity:.2f}', ha='center', va='bottom', fontsize=10)
plt.show()

# 그래프 그리기 (popularity 및 score 기준)
plt.figure(figsize=(14, 8))
ax3 = top_10_movies_popularity['popularity'].plot(kind='bar', color='lightgreen', position=0, width=0.4, align='center')
ax4 = ax3.twinx()
top_10_movies_popularity['score'].plot(kind='bar', color='skyblue', position=1, width=0.4, align='center', ax=ax4)

for i, (popularity, score) in enumerate(zip(top_10_movies_popularity['popularity'], top_10_movies_popularity['score'])):
    ax3.text(i, popularity + 1, f'{popularity:.2f}', ha='center', va='bottom', fontsize=10, color='green')
    ax4.text(i, score + 0.01, f'{score:.2f}', ha='center', va='bottom', fontsize=10)

ax3.set_title('Top 10 Movies by Popularity with Weighted Score')
ax3.set_xlabel('Movie Title')
ax3.set_ylabel('Popularity')
ax4.set_ylabel('Weighted Score')
ax3.set_xticklabels(top_10_movies_popularity['title'], rotation=45)
ax3.legend(['Popularity'], loc='upper left')
ax4.legend(['Weighted Score'], loc='upper right')

plt.show()
