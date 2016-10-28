
# coding: utf-8

# In[130]:


import sys
import csv
import numpy as np

#train_file = 'spam_data/spam_train.csv'
train_file = sys.argv[1]

#model_file = 'model.csv'
model_file = sys.argv[2] + '.csv'

train_X_y = np.ndarray(shape=(4001,58))
train_0 = np.ndarray(shape=(2447,56))
train_1 = np.ndarray(shape=(1554,56))

#read the training data
f = open(train_file, 'r')
i = 0
for row in csv.reader(f):
    j = -1
    for data in row:
        if j < 0:
            j += 1
            continue
        train_X_y[i, j] = data
        j += 1
    i += 1
f.close()

train_X_y = train_X_y[np.argsort(train_X_y[:, -1])]

for i in range(train_X_y.shape[0]):
    if train_X_y[i, -1] == 1:
        train_0[:, :31] = train_X_y[:i, :31]
        train_0[:, 31:] = train_X_y[:i, 32:-1]
        train_1[:, :31] = train_X_y[i:, :31]
        train_1[:, 31:] = train_X_y[i:, 32:-1]
        #train_0 = train_X_y[:i, :-1]
        #train_1 = train_X_y[i:, :-1]
        break    
        
#calculate the mean and the std
mean_0 = np.mean(train_0, axis=0)
std_0 = np.std(train_0, axis=0)
mean_1 = np.mean(train_1, axis=0)
std_1 = np.std(train_1, axis=0)


# In[131]:

def getClassProb(x, mean, std):
    exponent = np.exp(-(np.power(x - mean, 2) / (2 * np.power(std, 2))))
    prob = (1 / (np.sqrt(2 * np.pi) * std)) * exponent
    return np.prod(prob, axis=1)

def predict(x, prob_0, prob_1):
    label = np.ndarray(shape=(x.shape[0], 1))
    for i in range(x.shape[0]):
        if prob_0[i] > prob_1[i]:
            label[i] = 0
        else:
            label[i] = 1
    return label

def getAccu(y, label):
    correct = 0.
    for i in range(y.shape[0]):
        if label[i] == y[i]:
            correct += 1
    return correct / y.shape[0]


# In[132]:

train_X = np.ndarray(shape=(4001,56))
train_X[:, :31] = train_X_y[:, :31]
train_X[:, 31:] = train_X_y[:, 32:-1]
train_y = train_X_y[:, -1]

train_prob_0 = getClassProb(train_X, mean_0, std_0)
train_prob_1 = getClassProb(train_X, mean_1, std_1)

train_label = predict(train_X, train_prob_0, train_prob_1)
train_accu = getAccu(train_y, train_label)

print train_accu


# In[134]:

model = []
l = []
for i in range(mean_0.shape[0]):    
    l.append(str(mean_0[i]))    
model.append(l)
l = []
for i in range(std_0.shape[0]):    
    l.append(str(std_0[i]))
model.append(l)
l = []
for i in range(mean_1.shape[0]):
    l.append(str(mean_1[i]))
model.append(l)
l = []
for i in range(std_1.shape[0]):
    l.append(str(std_1[i]))
model.append(l)
f = open(model_file, 'w')
w = csv.writer(f)
w.writerows(model)
f.close()


# In[ ]:



