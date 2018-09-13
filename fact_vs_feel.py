from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
import random
import os
from sklearn.metrics import accuracy_score
import numpy as np

data = []
data_labels = []

# get the fact data
for filename in os.listdir("/Users/andrei/Desktop/data-fact-feel-v2/train/fact"):
    if filename.endswith("txt"): 
        path = '/Users/andrei/Desktop/data-fact-feel-v2/train/fact/' + filename
        with open(path, 'r') as content_file:
            content = content_file.read()
            data.append(content)
            data_labels.append('fact')

for filename in os.listdir("/Users/andrei/Desktop/data-fact-feel-v2/test/fact"):
    if filename.endswith("txt"): 
        path = '/Users/andrei/Desktop/data-fact-feel-v2/test/fact/' + filename
        with open(path, 'r') as content_file:
            content = content_file.read()
            data.append(content)
            data_labels.append('fact')
        
# get the feel data
for filename in os.listdir("/Users/andrei/Desktop/data-fact-feel-v2/train/feel"):
    if filename.endswith("txt"): 
        path = '/Users/andrei/Desktop/data-fact-feel-v2/train/feel/' + filename
        with open(path,'r') as content_file:
            content = content_file.read()
            data.append(content)
            data_labels.append('feel')

for filename in os.listdir("/Users/andrei/Desktop/data-fact-feel-v2/test/feel"):
    if filename.endswith("txt"): 
        path = '/Users/andrei/Desktop/data-fact-feel-v2/test/feel/' + filename
        with open(path,'r') as content_file:
            content = content_file.read()
            data.append(content)
            data_labels.append('feel')
        
        
vectorizer = CountVectorizer(
    analyzer = 'word',
    lowercase = False,
)
features = vectorizer.fit_transform(
    data
)
features_nd = features.toarray() # for easy usage

X_train, X_test, y_train, y_test  = train_test_split(
        features_nd, 
        data_labels,
        train_size=0.80, 
        random_state=1234)

log_model = LogisticRegression()

log_model = log_model.fit(X=X_train, y=y_train)

y_pred = log_model.predict(X_test)

print("Examples:")
text1 = "You should be proud of yourself"
text2 = "The lab coat is white"
new_data = [text1,text2]
print(new_data)
print(log_model.predict(vectorizer.transform(new_data).toarray()))



print("Accuracy:")

print(accuracy_score(y_test, y_pred))

print("Done")

