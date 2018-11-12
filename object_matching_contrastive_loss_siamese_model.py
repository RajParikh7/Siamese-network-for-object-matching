# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 16:27:30 2018

@author: rajpa
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 21:03:26 2018

@author: rajpa
"""

from keras.models import Sequential,Model
from keras.layers import Input,BatchNormalization
from keras.layers import Conv2D
from keras.layers import Dropout
from keras.layers import MaxPool2D
from keras.layers import Dense,Softmax,Flatten,Lambda
from keras.optimizers import adam
from keras.activations import softmax
import random
import numpy as np
import random
from keras.models import load_model
from sklearn.preprocessing import OneHotEncoder
from collections import defaultdict

def class_creator(train_x,train_y):
    classes=dict()
    for i in range(len(train_x)):
        if train_y[i] in classes:
            classes[train_y[i]].append(train_x[i])
        else:
            classes[train_y[i]]=[train_x[i]]
    return classes

def pair_create(classes):
    pairs=[]
    labels=[]
    for x in range(len(classes)):
        for i in range(len(classes[x])):
            k=random.randint(1,4999)
            j=random.randint(1,9)
            while i==j:
                j=random.randint(0,9)
                
            pairs.append([classes[x][i],classes[x][i-k]])
            pairs.append([classes[x][i],classes[j][k]])
            labels.append(1)
            labels.append(0)
    return pairs,labels

def Model1():
    input1=Input(shape=(32,32,3))
    conv1=Conv2D(filters=32,kernel_size=5, activation='relu',padding='same')(input1)
    pool1=MaxPool2D(pool_size=(2,2))(conv1)
    pool1=BatchNormalization()(pool1)
    conv2=Conv2D(filters=32,kernel_size=5, activation='relu',padding='same')(pool1)
    pool2=MaxPool2D(pool_size=(2,2))(conv2)
    pool2=BatchNormalization()(pool2)
    conv3=Conv2D(filters=64,kernel_size=5, activation='relu',padding='same')(pool2)
    pool3=MaxPool2D(pool_size=(2,2))(conv3)
    pool3=BatchNormalization()(pool3)
    flat=Flatten()(pool3)
    fc1=Dense(256,activation='relu')(flat)
    fc1=BatchNormalization()(fc1)
    fc2=Dense(128,activation='relu')(fc1)
    return Model(input1,fc2)



def distance(vector):
    from keras import backend as k
    import tensorflow as tf
    vector_a,vector_p= vector[0],vector[1]
    #print(vector_a)
    #print(vector_p)
    
    sum1=k.sum(k.square(vector_a-vector_p),axis=1)
    sum1=k.sqrt(sum1)
    #the tensors are reshaped as tensorflow treats [-1,] (tf constant) differently from [-1,1] tf 1d variable
    sump=tf.reshape(sum1,[-1,1])
    #print(sum1)
    #margin as added to ensure min difference between positive and negative, else the loss would be zero even for all vectors being zero
    return sump

def contrastive_loss(labels,loss):
    from keras import backend as k
    #import tensorflow as tf
    #taking max between loss and 0 to ensure non negative loss
    margin=0.2
    
    loss=k.square(labels*loss) + k.square((1-labels)*k.max(margin-loss,0))
    
    return k.mean(loss)
def accuracy_siamese(labels,loss):
    from keras import backend as k
    
    counter=k.greater(loss,0)# values where euclidean distance between a and p + margin is greater than euclidean distance between a and n
    total=k.greater(loss,float('-Inf'))
    total_int=k.cast(total,'float32')
    counter_int=k.cast(counter,'float32')    
    total=k.sum(total_int)
    counter=k.sum(counter_int)
    counter=total-counter
    
    count_loss=k.greater(loss,0.2)
    count_label=k.greater(labels,0.5)
    match=k.equal(count_loss,count_label)
    total=k.greater(loss,float('-Inf'))
    total_int=k.cast(total,'float32')
    match_int=k.cast(match,'float32')
    
    return match_int/total_int
    
    
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict1 = pickle.load(fo, encoding='bytes')
    return dict1


#training phase 
model=Model1()


inputa=Input(shape=(32,32,3))
inputp=Input(shape=(32,32,3))

# creates feature vector for 3 different inputs but with weights being shared
vector_a=model(inputa)
vector_p=model(inputp)

loss=Lambda(distance)([vector_a,vector_p])
print(loss)  
model1=Model(inputs=[inputa,inputp],outputs=loss)



data1=unpickle('E:\cifar-10-python\cifar-10-batches-py\data_batch_1')
data2=unpickle('E:\cifar-10-python\cifar-10-batches-py\data_batch_2')
data3=unpickle('E:\cifar-10-python\cifar-10-batches-py\data_batch_3')
data4=unpickle('E:\cifar-10-python\cifar-10-batches-py\data_batch_4')
data5=unpickle('E:\cifar-10-python\cifar-10-batches-py\data_batch_5')

train_x=list(data1[b'data'])+list(data2[b'data'])+list(data3[b'data'])+list(data4[b'data'])+list(data5[b'data'])

train_x=np.array(train_x)
length=len(train_x)
r=train_x[:,:1024]
g=train_x[:,1024:2048]
b=train_x[:,2048:3072]
r=r.reshape(length,32,32)
g=g.reshape(length,32,32)
b=b.reshape(length,32,32) 
train_x=np.reshape(train_x,(length,32,32,3))
train_x[:,:,:,0]=r/255
train_x[:,:,:,1]=g/255
train_x[:,:,:,2]=b/255
#assert(train_x.shape==(10000,32,32,3))

enc = OneHotEncoder(handle_unknown='ignore')
train_y=data1[b'labels']+data2[b'labels']+data3[b'labels']+data4[b'labels']+data5[b'labels']
classes=class_creator(train_x,train_y)

pairs,labels=pair_create(classes)
pairs=np.array(pairs)
labels=np.array(labels)
train_y=[[i] for i in train_y]
'''
train_y=enc.fit_transform(train_y)
train_y=train_y
train_y_dummy=[0]*len(pairs)
train_y_dummy=np.array(train_y_dummy)
'''
#train_y=np.array(train_y)
#print(train_y.shape)
#assert(train_y.shape==(10000,10))
#model1=load_model('Siamese2.h5',custom_objects={'contrastive_loss': contrastive_loss,'accuracy_siamese':accuracy_siamese})

adam1=adam(lr=0.000001)
model1.compile(optimizer=adam1,
              loss=contrastive_loss,
              metrics=[accuracy_siamese]
              ) 
details=model1.fit([pairs[:,0],pairs[:,1]],labels,batch_size=16,epochs=5)
model1.save('Siamese2.h5')


#inference phase


test_x=list(pairs[:100])+list(pairs[20000:20100])
test_y=list(labels[:100])+list(labels[20000:20100])
test_x,test_y=np.array(test_x),np.array(test_y)
#test_class=class_creator(train_x,train_y)  

#test_pair,test_out=pair_create(test_class)
test_loss=model1.predict([pairs[:,0],pairs[:,1]])
