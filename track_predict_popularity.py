#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
print(tf.__version__)


# In[24]:


# import tensorflow as tf
# from tensorflow.keras import layers, models
# from tensorflow.keras.applications.vgg16 import VGG16
# from sklearn.utils import shuffle
# from sklearn.model_selection import train_test_split

import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics


# In[3]:


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Restrict TensorFlow to only use the fourth GPU
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')

        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


# In[4]:


df = pd.read_csv("tracks.csv")
df


# In[5]:


df['release_date']=pd.to_datetime(df['release_date']).dt.year
df


# In[6]:


df = df[['popularity', 'duration_ms', 'release_date', 'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature']]
df


# In[7]:


# popularity = df.pop('popularity')


# In[9]:


x = df.iloc[:, 1:]  # The first to second-last columns are the features
y = df.iloc[:, 0]   # The last column is the ground-truth label
print(np.unique(y))
print(x.shape)


# In[10]:


X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.20, random_state=42)


# In[11]:


sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)


# In[12]:


print(X_train_std.shape)
print(X_test_std.shape)
print(y_train.shape)
print(y_test.shape)


# In[13]:


dataset = tf.data.Dataset.from_tensor_slices((X_train_std, y_train))


# In[14]:


for feat, targ in dataset.take(5):
    print ('Features: {}, Target: {}'.format(feat, targ))


# In[15]:


train_dataset = dataset.shuffle(len(df)).batch(64)


# In[16]:


def get_compiled_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(14, activation='relu'),
        tf.keras.layers.Dense(28, activation='relu'),
        tf.keras.layers.Dense(28, activation='relu'),
        tf.keras.layers.Dense(14, activation='relu'),
        tf.keras.layers.Dense(1)
    ])

    model.compile(optimizer='adam',
                loss=tf.keras.losses.MeanSquaredError(),
                metrics=['MeanAbsoluteError'])
    return model


# In[17]:


model = get_compiled_model()
model.fit(train_dataset, epochs=15)


# In[18]:


predict = model(X_test_std).numpy()[:,0]

y_test_v = y_test.values

diff = abs(y_test_v-predict)
print(diff)


# In[19]:


import matplotlib.pyplot as plt
plt.xlabel("data_num", fontsize=14)
plt.ylabel("popularuty", fontsize=14)

# plt.plot(predict)
plt.plot(diff)

plt.legend()
plt.show()


# In[20]:


with open('your_file.txt', 'w') as f:
    for item in diff:
        f.write("%s\n" % item)


# In[21]:


diff.shape


# In[22]:


model.evaluate(X_test_std, y_test.values)


# In[29]:


metrics.mean_absolute_error(y_test_v,predict)


# In[ ]:




