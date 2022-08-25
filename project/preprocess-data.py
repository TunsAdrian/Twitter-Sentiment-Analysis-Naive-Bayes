import numpy as np
import pandas as pd
import json
import re
from nltk.stem.wordnet import WordNetLemmatizer


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
        if word.lower() in contractions:
            tweet = tweet.replace(word, contractions[word.lower()])  # Replaces contractions
    tweet = re.sub(r"[^\w\s]", " ", str(tweet))  # Removes all punctuations
    tweet = re.sub(r'(.)\1+', r'\1\1', tweet)  # Convert more than 2 letter repetitions to 2 letter
    tweet = re.sub(r"\s+", " ", str(tweet))  # Replaces double spaces with single space
    return tweet


if __name__ == '__main__':
    stop_words = {"i", "me", "my", "myself", "we", "our", "ours", "ourselves",
                  "you", "your", "yours", "yourself", "yourselves", "he", "him",
                  "his", "himself", "she", "her", "hers", "herself", "it", "its",
                  "itself", "they", "them", "their", "theirs", "themselves", "what",
                  "which", "who", "whom", "this", "that", "these", "those", "am", "is",
                  "are", "was", "were", "be", "been", "being", "have", "has", "had",
                  "having", "do", "does", "did", "doing", "a", "an", "the", "and",
                  "but", "if", "or", "because", "as", "until", "while", "of", "at",
                  "by", "for", "with", "about", "against", "between", "into", "through",
                  "during", "before", "after", "above", "below", "to", "from", "up",
                  "down", "in", "out", "on", "off", "over", "under", "again", "further",
                  "then", "once", "here", "there", "when", "where", "why", "how", "all",
                  "any", "both", "each", "few", "more", "most", "other", "some", "such",
                  "only", "own", "same", "so", "than", "too", "very",
                  "can", "will", "just", "should", "now"}

    with open('../assets/contractions.json', 'r') as f:
        contractions_dict = json.load(f)

    contractions = contractions_dict['contractions']
    pd.set_option('display.max_colwidth', None)

    total_data = pd.read_csv("../dataset/tweets.csv", encoding="ISO-8859-1")

    total_data['processed_tweet'] = np.vectorize(process_tweet)(total_data['SentimentText'])

    tokenized_tweet = total_data['processed_tweet'].apply(lambda x: x.split())

    lemmatizer = WordNetLemmatizer()
    tokenized_tweet = tokenized_tweet.apply(lambda x: [lemmatizer.lemmatize(i) for i in x])

    for i in range(len(tokenized_tweet)):
        tokenized_tweet[i] = ' '.join([word for word in tokenized_tweet[i] if word not in stop_words])

    total_data['processed_tweet'] = tokenized_tweet

    total_data.to_csv('../dataset/processed-tweets.csv', index=False)
