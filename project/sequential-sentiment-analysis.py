import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import time

total_data = pd.read_csv('../dataset/processed-tweets.csv')

start_time = time.time()

vectorizer = TfidfVectorizer()
final_vectorized_data = vectorizer.fit_transform(total_data['processed_tweet'].apply(lambda x: np.str_(x)))

X_train, X_test, y_train, y_test = train_test_split(final_vectorized_data, total_data['Sentiment'], test_size=0.2, random_state=69)

model_naive = MultinomialNB().fit(X_train, y_train)
predicted_naive = model_naive.predict(X_test)

score_naive = accuracy_score(y_test, predicted_naive)
print('Accuracy of Naive-bayes: ', score_naive)

print('Execution time: %s seconds' % (time.time() - start_time))
