<<<<<<< HEAD
=======
#this is basically checking different classifiers which we can use ,like decision tree,knaive bayes,random forest and adaaboost classifiers


>>>>>>> 2e2133a456bb3c27032ac46fdbd632c9f01b2577
#!/usr/bin/env python
# coding: utf-8

# # Movie Reviews using different Classifiers :

# In[3]:


import numpy as np 
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import nltk
import re
import string
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn import metrics

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import AdaBoostClassifier


# In[18]:


# Checking for null values in train and test datasets
print("Train Dataset - Null Values:")
print(train_csv.isnull().sum())

print("\nTest Dataset - Null Values:")
print(test_csv.isnull().sum())


# In[19]:


# Descriptive statistics
print("Descriptive Statistics:")
print(train_csv.describe())

# Class distribution
print("\nClass Distribution:")
print(train_csv['1'].value_counts())

# Review length distribution
train_csv['Review_Length'] = train_csv['0'].apply(lambda x: len(x.split()))
print("\nReview Length Distribution:")
print(train_csv['Review_Length'].describe())


# ## Using tf-idf :

# In[4]:


test_csv = pd.read_csv('test_data (1).csv')
train_csv = pd.read_csv('train_data (1).csv')

train_X = train_csv['0']   # '0' corresponds to Texts/Reviews
train_y = train_csv['1']   # '1' corresponds to Label (1 - positive and 0 - negative)
test_X = test_csv['0']
test_y = test_csv['1']


# In[5]:


tf_vectorizer = CountVectorizer()
X_train_tf = tf_vectorizer.fit_transform(train_X)
print("n_samples: %d, n_features: %d" % X_train_tf.shape)


# In[6]:


print(X_train_tf.toarray())


# In[7]:


X_test_tf = tf_vectorizer.transform(test_X)
print("n_samples: %d, n_features: %d" % X_test_tf.shape)


# ### Using naive bayse classifier :

# ##### For traing the model :

# In[8]:


naive_bayes_classifier = MultinomialNB()

naive_bayes_classifier.fit(X_train_tf, train_y)


# #### While testing the model :

# In[9]:


y_pred = naive_bayes_classifier.predict(X_test_tf)

score1 = metrics.accuracy_score(test_y, y_pred)
print("accuracy:   %0.3f" % score1)

print(metrics.classification_report(test_y, y_pred, target_names=['Positive', 'Negative']))

print("confusion matrix:")
print(metrics.confusion_matrix(test_y, y_pred))


# ### Using decision tree classifier :
# 

# In[10]:


count_vectorizer = CountVectorizer()
X_train_counts = count_vectorizer.fit_transform(train_X)
X_test_counts = count_vectorizer.transform(test_X)

decision_tree_classifier = DecisionTreeClassifier()
decision_tree_classifier.fit(X_train_counts, train_y)
y_pred = decision_tree_classifier.predict(X_test_counts)

accuracy = metrics.accuracy_score(test_y, y_pred)
print("Accuracy: %0.3f" % accuracy)

print(metrics.classification_report(test_y, y_pred, target_names=['Negative', 'Positive']))

print("Confusion Matrix:")
print(metrics.confusion_matrix(test_y, y_pred))


# ### Using  Random forest classifier  :
# 

# In[11]:


random_forest_classifier = RandomForestClassifier()
random_forest_classifier.fit(X_train_counts, train_y)
y_pred = random_forest_classifier.predict(X_test_counts)

accuracy = metrics.accuracy_score(test_y, y_pred)
print("Accuracy: %0.3f" % accuracy)

print(metrics.classification_report(test_y, y_pred, target_names=['Negative', 'Positive']))

print("Confusion Matrix:")
print(metrics.confusion_matrix(test_y, y_pred))


# ### Using  adaboost classifier :
# ###### Taking base estimator as Naive Bayes

# In[12]:


base_estimator = MultinomialNB()

# Initialize AdaBoost classifier
adaboost_classifier = AdaBoostClassifier(base_estimator=base_estimator)

# Train the classifier
adaboost_classifier.fit(X_train_counts, train_y)

# Make predictions
y_pred = adaboost_classifier.predict(X_test_counts)

# Evaluate the model
accuracy = metrics.accuracy_score(test_y, y_pred)
print("Accuracy: %0.3f" % accuracy)

print(metrics.classification_report(test_y, y_pred, target_names=['Negative', 'Positive']))

print("Confusion Matrix:")
print(metrics.confusion_matrix(test_y, y_pred))


# ## K-fold :
# ##### Used to cross-validation to evaluate model performance more reliably.

# In[13]:


from sklearn.model_selection import cross_val_score

# Perform cross-validation for Naive Bayes classifier
scores = cross_val_score(naive_bayes_classifier, X_train_counts, train_y, cv=5)
print("Cross-Validation Scores for Naive Bayes:", scores)
print("Mean Accuracy:", scores.mean())


# # Real time Prediction :

# In[14]:


def predict_sentiment(review, model, vectorizer):
    review_counts = vectorizer.transform([review])
    prediction = model.predict(review_counts)
    return "Positive" if prediction == 1 else "Negative"

# Example usage
new_review = "I absolutely loved this movie! The acting was superb and the storyline was captivating."
print("Predicted Sentiment:", predict_sentiment(new_review, naive_bayes_classifier, count_vectorizer))


# # Visualizations: 

# In[15]:


from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(20,10))
plot_tree(decision_tree_classifier, filled=True, feature_names=count_vectorizer.get_feature_names_out())
plt.show()


# In[16]:


import matplotlib.pyplot as plt

# Plot class distribution
plt.figure(figsize=(8, 5))
train_y.value_counts().plot(kind='bar', color=['skyblue', 'salmon'])
plt.title('Class Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.xticks([0, 1], ['Negative', 'Positive'], rotation=0)
plt.show()


# In[20]:


pip install wordcloud


# In[21]:


from wordcloud import WordCloud

# Join all positive and negative reviews
positive_reviews = ' '.join(train_X[train_y == 1])
negative_reviews = ' '.join(train_X[train_y == 0])

# Generate word clouds
positive_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(positive_reviews)
negative_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(negative_reviews)

# Plot word clouds
plt.figure(figsize=(15, 6))
plt.subplot(1, 2, 1)
plt.imshow(positive_wordcloud, interpolation='bilinear')
plt.title('Positive Reviews Word Cloud')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(negative_wordcloud, interpolation='bilinear')
plt.title('Negative Reviews Word Cloud')
plt.axis('off')

plt.show()


# In[22]:


# Calculate review lengths
review_lengths = train_X.apply(lambda x: len(x.split()))

# Plot review length distribution
plt.figure(figsize=(8, 5))
plt.hist(review_lengths, bins=30, color='skyblue', edgecolor='black')
plt.title('Review Length Distribution')
plt.xlabel('Review Length')
plt.ylabel('Frequency')
plt.show()


# In[23]:


from collections import Counter

# Tokenize and count words in positive and negative reviews
positive_words = ' '.join(train_X[train_y == 1]).split()
negative_words = ' '.join(train_X[train_y == 0]).split()

# Calculate word frequencies
positive_word_freq = Counter(positive_words)
negative_word_freq = Counter(negative_words)

# Print most common words
print("Top 10 Most Common Words in Positive Reviews:", positive_word_freq.most_common(10))
print("Top 10 Most Common Words in Negative Reviews:", negative_word_freq.most_common(10))


# In[ ]:





# In[ ]:




