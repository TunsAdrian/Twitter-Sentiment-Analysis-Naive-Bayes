from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import CountVectorizer, IDF
from pyspark.ml import Pipeline
import time

spark = SparkSession.builder.appName('Twitter Sentiment Analysis').master('local[*]').getOrCreate()

df = spark.read.csv('../dataset_small/processed-tweets.csv', header=True, inferSchema=True)
# df = spark.read.csv('../dataset_large/processed-tweets-1600000.csv', header=True, inferSchema=True)
# df = spark.read.csv('hdfs:///project/processed-tweets.csv', header=True, inferSchema=True)
# df = spark.read.csv('hdfs:///project/processed-tweets-1600000.csv', header=True, inferSchema=True)

df = df.na.drop()

start_time = time.time()

tokenizer = Tokenizer(inputCol='processed_tweet', outputCol='raw_words')
vectorizer = CountVectorizer(inputCol=tokenizer.getOutputCol(), outputCol='raw_features')
idf = IDF(inputCol=vectorizer.getOutputCol(), outputCol='features')

pipeline = Pipeline(stages=[tokenizer, vectorizer, idf])
rescaled_data = pipeline.fit(df).transform(df)

train_df, test_df = rescaled_data.randomSplit([0.8, 0.2], seed=69)

nb = NaiveBayes(modelType='multinomial', labelCol='Sentiment')
nb_model = nb.fit(train_df)

predictions_df = nb_model.transform(test_df)
print('Execution time: %s seconds' % (time.time() - start_time))

evaluator = MulticlassClassificationEvaluator(labelCol='Sentiment', metricName='accuracy')
nb_accuracy = evaluator.evaluate(predictions_df)
print('Accuracy of Naive Bayes: ' + str(nb_accuracy))
