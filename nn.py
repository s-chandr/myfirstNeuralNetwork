import tensorflow as tf 
from tensorflow import keras 
import matplotlib.pyplot as plt 
%matplotlib inline 
import numpy as np
import pandas as pd
import seaborn as sns
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
train.head
# df = df.loc[:, 'pixel0':'pixel783']
print(train.shape) #with label 
print(test.shape) #without label
train_y = train['label'].astype('float32')
train_x = train.drop(['label'],axis=1).astype('int32')
test_x = test.astype('float32')
train_x = train_x/255 #scaling 
test_x =  test_x/255
print(train_x.shape)
print(test_x.shape)
model = keras.Sequential([
    keras.layers.Dense(100, input_shape = (784,), activation = 'relu'),
    keras.layers.Dense(10, activation = 'sigmoid')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
) 
model.fit( train_x , train_y , epochs = 5 )
results = model.predict(test_x)
results = np.argmax(results, axis = 1)
results = pd.Series(results, name = "Label")

submission = pd.concat([pd.Series(range(1,28001) , name = "ImageId"),results],axis = 1 )
submission.to_csv("submission.csv",index=False)
