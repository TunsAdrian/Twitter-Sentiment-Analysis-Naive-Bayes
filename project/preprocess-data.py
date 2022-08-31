import numpy as np
import pandas as pd
import json
import re
import time
from nltk.stem.wordnet import WordNetLemmatizer

stop_words = {"i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself",
              "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself",
              "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these",
              "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do",
              "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while",
              "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before",
              "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again",
              "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each",
              "few", "more", "most", "other", "some", "such", "only", "own", "same", "so", "than", "too", "very", "can",
              "will", "just", "should", "now"}

with open('../assets/contractions.json', 'r') as f:
    contractions = json.load(f)

total_data = pd.read_csv("../dataset_small/tweets.csv", encoding="ISO-8859-1")
# total_data = pd.read_csv("../dataset_large/training.1600000.new.csv", encoding="ISO-8859-1")


def emoji(tweet):
    # Smile -- :), : ), :-), (:, ( :, (-:, :') , :O
    tweet = re.sub(r'(:\s?\)|:-\)|\(\s?:|\(-:|:\'\)|:O)', ' positiveemoji ', tweet)
    # Laugh -- :D, : D, :-D, xD, x-D, XD, X-D
    tweet = re.sub(r'(:\s?D|:-D|x-?D|X-?D)', ' positiveemoji ', tweet)
    # Love -- <3, :*
    tweet = re.sub(r'(<3|:\*)', ' positiveemoji ', tweet)
    # Wink -- ;-), ;), ;-D, ;D, (;,  (-; , @-)
    tweet = re.sub(r'(;-?\)|;-?D|\(-?;|@-\))', ' positiveemoji ', tweet)
    # Sad -- :-(, : (, :(, ):, )-:, :-/ , :-|
    tweet = re.sub(r'(:\s?\(|:-\(|\)\s?:|\)-:|:-/|:-\|)', ' negetiveemoji ', tweet)
    # Cry -- :,(, :'(, :"(
    tweet = re.sub(r'(:,\(|:\'\(|:"\()', ' negetiveemoji ', tweet)

    return tweet


def process_tweet(tweet):
    tweet = tweet.lower()  # Lowercase the string
    tweet = re.sub('@[^\s]+', '', tweet)  # Removes usernames
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', ' ', tweet)  # Remove URLs
    tweet = re.sub(r"\d+", " ", str(tweet))  # Removes all digits
    tweet = re.sub('&quot;', " ", tweet)  # Remove (&quot;)
    tweet = emoji(tweet)  # Replaces Emojis
    tweet = re.sub(r"\b[a-zA-Z]\b", "", str(tweet))  # Removes all single characters
    for word in tweet.split():
        if word in contractions:
            tweet = tweet.replace(word, contractions[word])  # Replaces contractions
    tweet = re.sub(r"[^\w\s]", " ", str(tweet))  # Removes all punctuations
    tweet = re.sub(r'(.)\1+', r'\1\1', tweet)  # Convert more than 2 letter repetitions to 2 letter
    tweet = re.sub(r'\s+', " ", str(tweet))  # Replaces double spaces with single space

    return tweet


start_time = time.time()

# apply function to each entry and vectorize for faster looping
total_data['processed_tweet'] = np.vectorize(process_tweet)(total_data['SentimentText'])

# split the sentences
tokenized_tweets = total_data['processed_tweet'].apply(lambda x: x.split())

# bring the words to the root form
lemmatizer = WordNetLemmatizer()
tokenized_tweets = tokenized_tweets.apply(lambda x: [lemmatizer.lemmatize(i) for i in x])

# remove the stop words and rebuild the sentences
for i in range(len(tokenized_tweets)):
    tokenized_tweets[i] = ' '.join([word for word in tokenized_tweets[i] if word not in stop_words])

total_data['processed_tweet'] = tokenized_tweets
print('Execution time: %s seconds' % (time.time() - start_time))

total_data.to_csv('../dataset_small/processed-tweets.csv', index=False)
# total_data.to_csv('../dataset_large/processed-tweets-1600000.csv', index=False)

print('Program finished successfully')
