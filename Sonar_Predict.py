
# coding: utf-8

# In[96]:


#get_ipython().magic(u'config IPCompleter.greedy=True')


# In[97]:


import matplotlib.pyplot as plt


# In[98]:


import tensorflow as tf


# In[99]:


import numpy as np


# In[100]:


import pandas as pd


# In[101]:


from sklearn.preprocessing import LabelEncoder


# In[102]:


from sklearn.utils import shuffle


# In[103]:


from sklearn.model_selection import train_test_split


# In[104]:


#set debug
#from IPython.core.debugger import set_trace

def one_hot_encode(labels):
    n_labels=len(labels)
    #set_trace() #debug
    n_unique_labels=len(np.unique(labels))
    one_hot_encode =np.zeros((n_labels,n_unique_labels))
    one_hot_encode[np.arange(n_labels), labels]=1
    return one_hot_encode
def read_DataSet(fileName):
    df = pd.read_csv(fileName)
    #last column is label
    rows,cols = df.shape
    X = df[df.columns[0:cols-1]].values
    Y = df[df.columns[cols-1]]
    encoder = LabelEncoder()
    encoder.fit(Y)
    Y = encoder.transform(Y)
    coded_Y = one_hot_encode(Y)
    
    print(X.shape)
    return (X,coded_Y)


# In[105]:


dataFile="sonar.all-data.csv"


# In[106]:


X,Y=read_DataSet(dataFile);


# In[107]:


X,Y = shuffle(X,Y,random_state=1)


# In[108]:


train_x,test_x,train_y,test_y = train_test_split(X,Y,test_size=0.2, random_state=200)


# In[109]:


print("training set x shape",(train_x.shape))
print("training set y shape",train_y.shape)
print("testing set x shape", test_x.shape)


# In[110]:


learning_rate=0.3
training_epoches=10
cost_history=np.empty(shape=[1], dtype=float)
n_dim=X.shape[1]
print ("dimension {0}".format(n_dim))
n_class=2
model_path="NMI"


# In[111]:


#define number of hidden layer
n_hidden_1=60
n_hidden_2=60
n_hidden_3=60
n_hidden_4=60


# In[112]:


#define the input parameters with column vector n_dim as dimesion
x = tf.placeholder(tf.float32, [None,n_dim])
W = tf.Variable(tf.zeros([n_dim,n_class]))
b = tf.Variable(tf.zeros([n_dim]))
real_y = tf.placeholder(tf.float32, [None,n_class])


# In[113]:


#define the Model
def multilayer_perceptron ( x,weights, biases):
    #Hidden layer with RELU activation
    layer_1=tf.add(tf.matmul(x,weights['h1']),biases['b1'])
    layer_1 = tf.nn.sigmoid(layer_1)
    
    #Hidden layer with sigmoid activation
    layer_2 = tf.add(    tf.matmul(layer_1,weights['h2']),biases['b2'])
    layer_2 = tf.nn.sigmoid(layer_2)

    #Hidden layer with sigmoid activation
    layer_3 = tf.add(    tf.matmul(layer_2,weights['h3']),biases['b3'])
    layer_3 = tf.nn.sigmoid(layer_3)
    
    #Hidden layer with sigmoid activation
    layer_4 = tf.add(    tf.matmul(layer_3,weights['h4']),biases['b4'])
    layer_4 = tf.nn.relu(layer_4)
    
    #output layer with linear activation
    #out_layer = tf.add(tf.matmul(layer_4,weights['out']),biases['out'])
    #out_layer = tf.nn.softmax(out_layer)
    out_layer = tf.matmul(layer_4,weights['out'])+biases['out']
    return out_layer
    


# In[114]:


weights={
    'h1':tf.Variable(    tf.truncated_normal([n_dim,n_hidden_1])),
    'h2':tf.Variable(    tf.truncated_normal([n_hidden_1,n_hidden_2])),
    'h3':tf.Variable(    tf.truncated_normal([n_hidden_2,n_hidden_3])),
    'h4':tf.Variable(    tf.truncated_normal([n_hidden_3,n_hidden_4])),
    'out':tf.Variable(    tf.truncated_normal([n_hidden_4,n_class]))
}
biases = {
    'b1':tf.Variable(    tf.truncated_normal([n_hidden_1])),
    'b2':tf.Variable(    tf.truncated_normal([n_hidden_2])),
    'b3':tf.Variable(    tf.truncated_normal([n_hidden_3])),
    'b4':tf.Variable(    tf.truncated_normal([n_hidden_4])),
    'out':tf.Variable(    tf.truncated_normal([n_class]))
}


# In[115]:


predicted_y = multilayer_perceptron(x,weights,biases)


# In[116]:


#implementation the linear lost function: SUM( real_y - predicted_y{activatefunc(linear weight dot product + bias)})/N
cost_function=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predicted_y, labels=real_y))
training_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)


# In[117]:


mse_history=[]
accuracy_history=[]


# In[120]:


#reload the saved model and run testing
init = tf.global_variables_initializer()

saver = tf.train.Saver()
sess = tf.Session()
sess.run(init)
saver.restore(sess,model_path)

#with tf.Session() as session:
prediction = tf.argmax(predicted_y,1)
correction_prediction = tf.equal(prediction, tf.argmax(real_y,1))
print("0 stands for M")
for i in range(93,101):
        prediction_run = sess.run(prediction,feed_dict={x:X[i].reshape(1,60)})
        accruacy_run=sess.run(correction_prediction,feed_dict={x:X[i].reshape(1,60), real_y:Y[i].reshape(1,2)})
        print("Original class:{0}, Predicted values:{1}".format(np.argmax(Y[i]),prediction_run))


# In[ ]:




