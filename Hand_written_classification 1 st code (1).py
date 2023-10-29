#!/usr/bin/env python
# coding: utf-8

# In[3]:


pip install tensorflow #installing tensorflow


# In[5]:


import tensorflow #importing tensorflow


# In[6]:


import tensorflow as tf  #importing tensorflow
from tensorflow import keras #importing keras
import matplotlib.pyplot as plt  #importing matplotlib for plotting
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np #importing nmpy


# In[7]:


(X_train, y_train) , (X_test, y_test) = keras.datasets.mnist.load_data() #imporing mnist dataset and dataset split


# In[8]:


len(X_train) #total number of elements in X_train


# In[9]:


len(X_test) #total number of elements in X_test


# In[10]:


X_train[0].shape # X_train consists of 60000 elements each of 28x28 matrices where each matrix corresponds to each element 


# In[11]:


X_train[0] # shows 28x28 matrix of 0th element in  X_train


# In[12]:


plt.matshow(X_train[0])
#plotting 0th element in X_train. consists of 60000 elements, each in 28x28 matrix format


# In[14]:


plt.matshow(X_test[0])
#plotting 0th element in X_test. consists of 10000 elements, each in 28x28 matrix format


# In[23]:


len(y_test) # y_test consists of 10000 elements, each in single colum


# In[25]:


y_test[0].shape # y_test consists of 10000 elements, each in single colum


# In[15]:


y_test[0] #shows 0th element in y_test. y_test consists of 10000  elements, 0th element is 7


# In[27]:


len(y_train)# y_train consists of 60000 elements, each in single colum


# In[28]:


y_train[0].shape # y_train consists of 60000 elements, each in single colum


# In[17]:


X_train = X_train / 255 #scaling to increase accracy
X_test = X_test / 255 #scaling to increase accracy


# In[18]:


X_train[0]


# In[19]:


X_train.shape


# In[20]:


X_train_flattened = X_train.reshape(len(X_train), 28*28)
X_test_flattened = X_test.reshape(len(X_test), 28*28)


# In[21]:


X_train_flattened.shape # X_train_flattened consists of single colums of 784 length and there are 60000 such single colums


# In[22]:


X_train_flattened[0] # X_train_flattened consists of single colums of 784 length and there are 60000 such single colums. This shows the first element


# In[30]:


model = keras.Sequential([
    keras.layers.Dense(10, input_shape=(784,), activation='sigmoid') #we choose Dense layer because only input layer of 784 and output layer of 10(0to9)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy', # we used sparse_categorical_crossentropy to calculate loss, other loss functions can be used
              metrics=['accuracy']) # we used accuracy to calculate accuracy, other accuracy functions can be used

model.fit(X_train_flattened, y_train, epochs=5) # fit() trains the data with X_train_flattened, y_train and epochs consists of iterations


# In[31]:


#when training, it is actually testing accruacy in training dataset


# In[32]:


model.evaluate(X_test_flattened, y_test) #testing accruacy in test dataset


# In[33]:


y_predicted = model.predict(X_test_flattened)
y_predicted[0] # we predict the result for input X_test_flattened and show y_predicted[0] we cannot predict single value we predict for whole data set first and check for each element


# In[39]:


len(y_predicted) #y_predicted consists of 10000 elements


# In[40]:


y_predicted.shape #y_predicted consists of 10000 elements each element consists of a single colum array of 10, where 10 corresponds to each outputs


# In[35]:


np.argmax(y_predicted[0]) #maximum predicted value for y_predicted[0] at output which is 7.  y_predicted[0]consists of output for output number 7 which means first element in 10000 corresponds to output 7


# In[38]:


np.argmax(y_predicted[1]) #maximum predicted value for y_predicted[1] at output which is 2. y_predicted[1]consists of output for output number 2 which means second element in 10000 single array y_predicted corresponds to output 7


# In[48]:


plt.matshow(X_test[0]) # checking corresponding input in X_test to verify


# In[50]:


plt.matshow(X_test[1])


# In[42]:


y_predicted_labels = [np.argmax(i) for i in y_predicted] # here we check the first 5 predicted values in y_predicted
y_predicted_labels[:5] ##ALSO y_predicted is in single colum array format, we convert it into digit for each array format


# In[55]:


for i in range(5):
    plt.matshow(X_test[i])
    plt.show() #To verify by plotting first 5 elements in X_test


# In[56]:


cm = tf.math.confusion_matrix(labels=y_test,predictions=y_predicted_labels) #we get confusion matrix named cm, so we assign predicted values on X axis and True values on Y axis
cm


# In[62]:


import seaborn as sn # importing seaborn for good image
plt.figure(figsize = (10,7)) # To adjust size of matrix
sn.heatmap(cm, annot=True, fmt='d') #create a heatmap of a confusion matrix (cm) with annotations. The annot=True argument tells Seaborn to display the numerical values inside the heatmap cells, and fmt='d' specifies the format of the displayed values as integers
plt.xlabel('Predicted')# assigning name to X axis
plt.ylabel('Truth')# assigning name to Y axis

#69.0: This is the x-coordinate (horizontal position) where the text will be placed.

#0.5: This is the y-coordinate (vertical position) where the text will be placed.

#'Truth': This is the text that will be displayed at the specified coordinates.


# In[63]:


model = keras.Sequential([
    keras.layers.Dense(100, input_shape=(784,), activation='relu'),#Here we give 100 neurons in hidden layer .Input 784, hidden layer 100, output neurons 10
    keras.layers.Dense(10, activation='sigmoid')#Also changing activation function for hidden and output layer. Also more layers, more accuracy
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train_flattened, y_train, epochs=5)


# In[66]:


#Using Flatten layer so that we don't have to call .reshape on input dataset
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)), #flattening 28x28 array to 28*28=784 colum single array
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dense(10, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10)


# In[64]:


model.evaluate(X_test_flattened,y_test) # checking accuracy


# In[65]:


#Doing same as before for hidden layer
y_predicted = model.predict(X_test_flattened)
y_predicted_labels = [np.argmax(i) for i in y_predicted]
cm = tf.math.confusion_matrix(labels=y_test,predictions=y_predicted_labels)

plt.figure(figsize = (10,7))
sn.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')


# In[67]:


model.evaluate(X_test,y_test)


# In[ ]:




