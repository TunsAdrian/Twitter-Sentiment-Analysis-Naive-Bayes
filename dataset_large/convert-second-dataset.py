import pandas as pd

total_data = pd.read_csv("./training.1600000.processed.noemoticon.csv", encoding="ISO-8859-1")

# drop unnecessary columns
total_data = total_data.drop(total_data.columns[[1, 2, 3, 4]], axis=1)

# convert sentiment value
total_data.iloc[:, 0] = total_data.iloc[:, 0].apply(lambda x: 1 if x == 4 else 0)
total_data.columns = ['Sentiment', 'SentimentText']

total_data.to_csv('../dataset_large/training.1600000.new.csv', index=False)
