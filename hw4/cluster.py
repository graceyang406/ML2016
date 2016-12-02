
# coding: utf-8

# In[42]:

import sys

#directory = 'data/'
directory = sys.argv[1]
#output_file = 'output.csv'
output_file = sys.argv[2]


# In[43]:

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer, HashingVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans

lines = [line.rstrip('\n') for line in open(directory + 'title_StackOverflow.txt')]

"""
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(lines)

tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts)

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
"""

vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = vectorizer.fit_transform(lines)

svd = TruncatedSVD(20)
lsa = make_pipeline(svd, Normalizer(copy=False))

X_train_lsa = lsa.fit_transform(X_train_tfidf)

km = KMeans(n_clusters=70, max_iter=100, n_init=1).fit(X_train_lsa)


# In[44]:

import csv

ans = [['ID', 'Ans']]
f = open('data/check_index.csv', 'r')
for i, row in enumerate(csv.reader(f)):
    if i == 0:
        continue
    for j, data in enumerate(row):
        if j == 1:
            x_label = km.labels_[int(data)]
        elif j == 2:
            y_label = km.labels_[int(data)]
    if x_label == y_label:
        ans.append([str(i - 1), '1'])
    else:
        ans.append([str(i - 1), '0'])
f.close()

f = open(output_file, 'w')
w = csv.writer(f)
w.writerows(ans)
f.close()


# In[ ]:



