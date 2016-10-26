
# coding: utf-8

# In[14]:


import sys
import csv
import numpy as np

#model_file = 'model.csv'
model_file = sys.argv[1] + '.csv'

#test_file = 'spam_data/spam_test.csv'
test_file = sys.argv[2]

#prediction_file = 'spam_data/prediction.csv'
prediction_file = sys.argv[3]

W = np.ndarray(shape=(1, 57))
b = np.ndarray(shape=(1))

test_X = np.ndarray(shape=(600,57))
test_y = np.ndarray(shape=(600,1))

#read the model
f = open(model_file, 'r')
for row in csv.reader(f):
    j = 0
    for data in row:
        if j < 57:
            W[0, j] = data
        else:
            b[0] = data
        j += 1
f.close()

#read the testing data
f = open(test_file, 'r')
i = 0
for row in csv.reader(f):
    j = -1
    for data in row:
        if j < 0:
            j += 1
            continue
        test_X[i, j] = data
        j += 1
    i += 1
f.close()


# In[15]:

def safe_exp(x, maxval=500):
    return np.exp(x.clip(max=maxval))


# In[16]:

test_label = [['id', 'label']]

#testing
for i in range(test_X.shape[0]):
    f = 1 / (1 + safe_exp(-(np.sum(test_X[i] * W) + b)))
    test_y[i, 0] = round(f)
    l = [str(i + 1), str(int(test_y[i, 0]))]
    test_label.append(l)
    
f = open(prediction_file, 'w')
w = csv.writer(f)
w.writerows(test_label)
f.close()


# In[ ]:



