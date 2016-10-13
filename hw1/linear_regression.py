
# coding: utf-8

# In[1]:


import csv
import numpy as np

train_data = np.ndarray(shape=(18,5760))
train_X_y = np.ndarray(shape=(5652,18,10))
train_X_raw = np.ndarray(shape=(5652,18,9))
train_y = np.ndarray(shape=(5652,1))
test_X_raw = np.ndarray(shape=(240,18,9))
test_y = np.ndarray(shape=(240,1))

f = open('data/train.csv', 'r')
i = -1
for row in csv.reader(f):
    if i < 0:
        i += 1
        continue  
    j = -3
    for data in row:
        if j < 0:
            j += 1
            continue
        if data == 'NR':
            data = 0
        train_data[i % 18, i / 18 * 24 + j] = data
        j += 1
    i += 1
f.close()

f = open('data/test_X.csv', 'r')
i = 0
for row in csv.reader(f):
    j = -2
    for data in row:
        if j < 0:
            j += 1
            continue
        if data == 'NR':
            data = 0
        test_X_raw[i / 18, i % 18, j] = data
        j += 1
    i += 1
f.close()

for i in range(18):
    for j in range(5760):
        if j % 480 < 471:
            train_X_y[j / 480 * 471 + j % 480, i, 0 : 10] = train_data[i, j : j + 10]
            
np.random.shuffle(train_X_y)

for i in range(5652):
    for j in range(18):
        train_X_raw[i, j, 0 : 9] = train_X_y[i, j, 0 : 9]
    train_y[i, 0] = train_X_y[i, 9, 9]


# In[37]:

num_X = 162
train_X = np.ndarray(shape=(5652,num_X))
test_X = np.ndarray(shape=(240,num_X))

train_X = np.reshape(train_X_raw, (5652,162))
test_X = np.reshape(test_X_raw, (240,162))

val_X = train_X[3768 : 5652]
val_y = train_y[3768 : 5652]
tr_X = train_X[0 : 3768]
tr_y = train_y[0 : 3768]

order = 1
tr_X_2 = np.ndarray(shape=(tr_X.shape[0],num_X*order))
val_X_2 = np.ndarray(shape=(val_X.shape[0],num_X*order))
test_X_2 = np.ndarray(shape=(test_X.shape[0],num_X*order))

for i in range(order):
    tr_X_2[ : , num_X * i : num_X * (i + 1)] = tr_X ** (i + 1)
    val_X_2[ : , num_X * i : num_X * (i + 1)] = val_X ** (i + 1)
    test_X_2[ : , num_X * i : num_X * (i + 1)] = test_X ** (i + 1)

tr_X = tr_X_2
val_X = val_X_2
test_X = test_X_2
num_X *= order

W = np.random.normal(0, 0.01, (1, num_X))
b = np.random.normal(0, 0.01, 1)

#regularization
lam = 10000

#learning rate
eta = 0.5

#adagrad

dW_adagrad = np.zeros(shape=(1,num_X))
db_adagrad = 0

#adam
"""
m_W = np.zeros(shape=(1, num_X))
v_W = np.zeros(shape=(1, num_X))
m_b = 0
v_b = 0
beta1 = 0.9
beta2 = 0.999
eps = 1e-8
"""
tr_loss = []
val_loss = []


# In[38]:

for it in range(100001):
    tr_L = lam * np.sum(W ** 2)
    dW = lam * 2 * W
    db = 0
    for i in range(tr_X.shape[0]):  
        tr_L += (tr_y[i] - (np.sum(tr_X[i] * W) + b)) ** 2
        dW += 2 * (tr_y[i] - (np.sum(tr_X[i] * W) + b)) * (-tr_X[i])
        db += 2 * (tr_y[i] - (np.sum(tr_X[i] * W) + b))
    #validation
    val_L = lam * np.sum(W ** 2)
    for i in range(val_X.shape[0]):
        val_L += (val_y[i] - (np.sum(val_X[i] * W) + b)) ** 2
    #adagrad update
    
    dW_adagrad += dW ** 2
    db_adagrad += db ** 2
    W -= eta * dW / np.sqrt(dW_adagrad)
    b -= eta * db / np.sqrt(db_adagrad)
    
    #adam update
    """
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
    """
    if it % 100 == 0:
        tr_loss.append(tr_L[0])
        val_loss.append(val_L[0])
        print "Iteration", it, ": average train loss is", tr_L[0] / 3768, ", average val loss is", val_L[0] / 1884


# In[35]:

test_y_list = [['id', 'value']]
for i in range(test_X.shape[0]):
    test_y[i] = np.sum(test_X[i] * W) + b
    l = ['id_' + str(i), str(test_y[i, 0])]
    test_y_list.append(l)
    
f = open('data/linear_regression.csv', 'w')
wr = csv.writer(f)
wr.writerows(test_y_list)
f.close()


# In[ ]:



