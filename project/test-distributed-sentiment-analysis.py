from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import CountVectorizer, IDF
from pyspark.ml import Pipeline
import time, json
from pyspark.ml.feature import StopWordsRemover
from pyspark.sql.functions import lower, col, udf, regexp_replace, split
from pyspark.sql.types import IntegerType, StringType, ArrayType
from nltk.stem.wordnet import WordNetLemmatizer


def emoji(tweet):
    # Smile -- :), : ), :-), (:, ( :, (-:, :') , :O
    tweet = regexp_replace(tweet, r'(:\s?\)|:-\)|\(\s?:|\(-:|:\'\)|:O)', ' positiveemoji ')
    # Laugh -- :D, : D, :-D, xD, x-D, XD, X-D
    tweet = regexp_replace(tweet, r'(:\s?D|:-D|x-?D|X-?D)', ' positiveemoji ')
    # Love -- <3, :*
    tweet = regexp_replace(tweet, r'(<3|:\*)', ' positiveemoji ')
    # Wink -- ;-), ;), ;-D, ;D, (;,  (-; , @-)
    tweet = regexp_replace(tweet, r'(;-?\)|;-?D|\(-?;|@-\))', ' positiveemoji ')
    # Sad -- :-(, : (, :(, ):, )-:, :-/ , :-|
    tweet = regexp_replace(tweet, r'(:\s?\(|:-\(|\)\s?:|\)-:|:-/|:-\|)', ' negetiveemoji ')
    # Cry -- :,(, :'(, :"(
    tweet = regexp_replace(tweet, r'(:,\(|:\'\(|:"\()', ' negetiveemoji ')
    return tweet


def process_tweet(tweet):
    tweet = lower(tweet)  # Lowercase the string
    tweet = regexp_replace(tweet, '@[^\s]+', '')  # Removes usernames
    tweet = regexp_replace(tweet, '((www\.[^\s]+)|(https?://[^\s]+))', ' ')  # Remove URLs
    tweet = regexp_replace(str(tweet), r"\d+", " ", )  # Removes all digits
    tweet = regexp_replace(tweet, '&quot;', " ")  # Remove (&quot;)
    tweet = emoji(tweet)  # Replaces Emojis
    tweet = regexp_replace(str(tweet), r"\b[a-zA-Z]\b", "")  # Removes all single characters
    # for word in tweet.split():
    #     if word.lower() in contractions:
    #         tweet = tweet.replace(tweet, word, contractions[word.lower()])  # Replaces contractions
    tweet = regexp_replace(str(tweet), r"[^\w\s]", " ")  # Removes all punctuations
    tweet = regexp_replace(tweet, r'(.)\1+', r'\1\1')  # Convert more than 2 letter repetitions to 2 letter
    tweet = regexp_replace(str(tweet), r'\s+', " ")  # Replaces double spaces with single space
    return tweet


spark = SparkSession.builder.appName('Twitter Sentiment Analysis').master('local[*]').getOrCreate()

stop_words = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves",
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
              "can", "will", "just", "should", "now"]

with open('../assets/contractions.json', 'r') as f:
    contractions_dict = json.load(f)

contractions = contractions_dict['contractions']

df = spark.read.csv('../dataset_small/tweets.csv', header=True, inferSchema=True)
# df = spark.read.csv('../dataset_large/processed-tweets-1600000.csv', header=True, inferSchema=True)

# df = spark.read.csv('hdfs:///project/processed-tweets.csv', header=True, inferSchema=True)
# df = spark.read.csv('hdfs:///project/processed-tweets-1600000.csv', header=True, inferSchema=True)
df = df.na.drop()

processTweets = udf(lambda x: process_tweet(x), StringType())
df = df.withColumn('hmm', processTweets(col('SentimentText')))
# df.show(5, True)

lemmatizer = WordNetLemmatizer()
sparkLemmer = udf(lambda x: lemmatizer.lemmatize(x), StringType())
df = df.withColumn('hmmwow', sparkLemmer(col('hmm')))
# df.show()

# start_time = time.time()
# df = df.withColumn("tokens", split("hmmwow", "\\s+"))
# remover = StopWordsRemover(stopWords=stop_words, inputCol="tokens", outputCol="stop")

tokenizer = Tokenizer(inputCol='hmmwow', outputCol='raw_words')
vectorizer = CountVectorizer(inputCol=tokenizer.getOutputCol(), outputCol='raw_features')
idf = IDF(inputCol=vectorizer.getOutputCol(), outputCol='features')

pipeline = Pipeline(stages=[tokenizer,  vectorizer, idf])
rescaled_data = pipeline.fit(df).transform(df)

train_df, test_df = rescaled_data.randomSplit([0.8, 0.2], seed=69)

nb = NaiveBayes(modelType='multinomial', labelCol='Sentiment')
nb_model = nb.fit(train_df)

predictions_df = nb_model.transform(test_df)
# print('Execution time: %s seconds' % (time.time() - start_time))

predictions_df.show(5, True)

evaluator = MulticlassClassificationEvaluator(labelCol='Sentiment', metricName='accuracy')
nb_accuracy = evaluator.evaluate(predictions_df)
print('Accuracy of Naive Bayes: ' + str(nb_accuracy))
