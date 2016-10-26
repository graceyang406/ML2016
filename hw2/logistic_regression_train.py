
# coding: utf-8

# In[53]:


import sys
import csv
import numpy as np

#train_file = 'spam_data/spam_train.csv'
train_file = sys.argv[1]

#model_file = 'model.csv'
model_file = sys.argv[2] + '.csv'

train_X_y = np.ndarray(shape=(4001,58))
train_X = np.ndarray(shape=(2667,57))
train_y = np.ndarray(shape=(2667,1))
val_X = np.ndarray(shape=(1334,57))
val_y = np.ndarray(shape=(1334,1))

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

#shuffle the training data
np.random.shuffle(train_X_y)

#split the data into training data and validation data
train_X[:, :] = train_X_y[:2667, :57]
train_y[:, 0] = train_X_y[:2667, 57]
val_X[:, :] = train_X_y[2667:, :57]
val_y[:, 0] = train_X_y[2667:, 57]


# In[75]:

W = np.random.normal(0, 0.001, (1, 57))
b = np.random.normal(0, 0.001, 1)

#regularization
lam = 0

#learning rate
eta = 0.5
"""
#adagrad
dW_adagrad = np.zeros(shape=(1,57))
db_adagrad = 0
"""
#adam
m_W = np.zeros(shape=(1,57))
v_W = np.zeros(shape=(1,57))
m_b = 0
v_b = 0
beta1 = 0.9
beta2 = 0.999
eps = 1e-8

result = [['train_L', 'val_L', 'train_accu', 'val_accu']]


# In[ ]:

def safe_ln(x, minval=0.0000000001):
    return np.log(x.clip(min=minval))

def safe_exp(x, maxval=500):
    return np.exp(x.clip(max=maxval))


# In[76]:

for it in range(3000):
    train_L = lam * np.sum(W ** 2)
    val_L = lam * np.sum(W ** 2)
    dW = lam * 2 * W
    db = 0
    train_accu = 0
    val_accu = 0
    
    #training
    for i in range(train_X.shape[0]):
        f = 1 / (1 + safe_exp(-(np.sum(train_X[i] * W) + b)))
        train_L += -(train_y[i] * safe_ln(f) + (1 - train_y[i]) * safe_ln(1 - f))
        dW += -(train_y[i] - f) * train_X[i]
        db += -(train_y[i] - f)
        train_accu += train_y[i] * round(f) + (1 - train_y[i]) * (1 - round(f))
        
    #validation
    for i in range(val_X.shape[0]):
        f = 1 / (1 + safe_exp(-(np.sum(val_X[i] * W) + b)))
        val_L += -(val_y[i] * safe_ln(f) + (1 - val_y[i]) * safe_ln(1 - f))
        val_accu += val_y[i] * round(f) + (1 - val_y[i]) * (1 - round(f))
    """    
    #adagrad update
    dW_adagrad += dW ** 2
    db_adagrad += db ** 2
    W -= eta * dW / np.sqrt(dW_adagrad)
    b -= eta * db / np.sqrt(db_adagrad)
    """
    #adam update
    m_W = beta1 * m_W + (1 - beta1) * dW
    v_W = beta2 * v_W + (1 - beta2) * (dW ** 2)
    m_b = beta1 * m_b + (1 - beta1) * db
    v_b = beta2 * v_b + (1 - beta2) * (db ** 2)
    #bias correction
    m_W_hat = m_W / (1 - beta1 ** (it + 1))
    v_W_hat = v_W / (1 - beta2 ** (it + 1))
    m_b_hat = m_b / (1 - beta1 ** (it + 1))
    v_b_hat = v_b / (1 - beta2 ** (it + 1))
    W -= eta * m_W_hat / (np.sqrt(v_W_hat) + eps)
    b -= eta * m_b_hat / (np.sqrt(v_b_hat) + eps)
    
    if it % 100 == 0:
        l = [str(train_L[0] / 2667), str(val_L[0] / 1334), str(train_accu[0] / 2667), str(val_accu[0] / 1334)]
        result.append(l)
        print 'Iteration', it, ': train_L =', train_L[0] / 2667, ', val_L =', val_L[0] / 1334,               ', train_accu =', train_accu[0] / 2667, ', val_accu =', val_accu[0] / 1334


# In[81]:

model = []
for i in range(W.shape[1]):
    model.append(str(W[0, i]))
model.append(str(b[0]))
model = [model]

f = open(model_file, 'w')
w = csv.writer(f)
w.writerows(model)
f.close()


# In[77]:

"""
f = open('spam_data/result.csv', 'w')
w = csv.writer(f)
w.writerows(result)
f.close()
"""


# In[ ]:



