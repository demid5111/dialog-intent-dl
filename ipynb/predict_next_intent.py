#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os, time, random
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import keras
from keras.layers import Dense, Input, GlobalMaxPooling1D, Flatten, TimeDistributed, Bidirectional
from keras.layers import Conv1D, MaxPooling1D, Embedding, LSTM, Dropout, Activation, Permute
from keras import regularizers
from keras.models import Model, Sequential
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import TensorBoard
from keras import backend as K
#from keras.backend import permute_dimensions
from sklearn.model_selection import train_test_split
print(tf.__version__)
print(keras.__version__)

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

if tf.test.gpu_device_name():
    print('GPU found')
else:
    print("No GPU found")
    
with tf.device('/cpu:0'):
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
    c = tf.matmul(a, b)

with tf.Session() as sess:
    print(sess.run(c))


# In[ ]:


def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = tf.graph_util.convert_variables_to_constants(
            session, input_graph_def, output_names, freeze_var_names)
        return frozen_graph
    
def save_model(session, model, name):
    frozen_graph = freeze_session(session, output_names=[out.op.name for out in model.outputs])
    tf.train.write_graph(frozen_graph, "../models", "{}".format(name), as_text=False)


# In[2]:


# from keras_handler.SequenceGenerator import SequenceGenerator
# sg = SequenceGenerator(PredictIntent.data_path, PredictIntent.intent_index, PredictIntent.max_sequence_length, 0.2)
# generator = sg.generate_batch(5, subset='training')
# tuple1 = next(generator)
# print(tuple1[0].shape)
# print(tuple1[1].shape)


# In[3]:


from keras_handler.predict_intent import PredictIntent


# In[4]:


pi = PredictIntent(is_general = False)
pi.batch_size = 13
pi.max_sequence_length = 5
pi.intent_embedding_dim = 10
pi.num_units = 30
pi.validation_split = 0.2
pi.random_state = 42
pi.data_path = "../feature_and_vector_seq"


# In[ ]:


pi.build_BiRNN_model()
history_BiRNN = pi.fit_generator(epochs = 100)
save_model(K.get_session(), history_BiRNN, 'history_BiRNN.pb')


# In[51]:


pi.build_RNN_model()
history_RNN = pi.fit_generator(epochs = 100)
save_model(K.get_session(), history_RNN, 'history_RNN.pb')


# In[56]:


pi.build_CNN_model()
history_CNN = pi.fit_generator(epochs = 100)
save_model(K.get_session(), history_CNN, 'history_CNN.pb')


# In[5]:


pi.build_MLP_model()
history_MLP = pi.fit_generator(epochs = 100)
save_model(K.get_session(), history_MLP, 'history_MLP.pb')


# In[88]:


len_y = 30
x = range(1,len_y+1)
y1 = history_BiRNN.history['val_acc'][:len_y]
y2 = history_RNN.history['val_acc'][:len_y]
y3 = history_CNN.history['val_acc'][:len_y]
y4 = history_MLP.history['val_acc'][:len_y]

plt.xlabel("Epochs")
plt.ylabel("Validation Accuracy")

plt.plot(x,y1, 'k-.', label='BiRNN', linewidth=2)
plt.plot(x,y2, 'k--', label='RNN', linewidth=1)
plt.plot(x,y3, 'k-', label='CNN', linewidth=1)
plt.plot(x,y4, 'k:', label='MLP', linewidth=2)
plt.legend(loc='upper right')


# In[ ]:


# 25 int

# RNN, 100 ep return_sequences=True 
# 13100 30 - loss: 2.9677 - acc: 0.1152
# 1310  30 - loss: 2.6313 - acc: 0.2464
# 131   30 - loss: 1.8456 - acc: 0.4678
# 13    30 - loss: 2.1301 - acc: 0.4263
# 1     30 - loss: 2.5108 - acc: 0.2834

# BiRNN, 100 ep return_sequences=True
# 1310    2s/step - loss: 2.1712 - acc: 0.3397 - val_loss: 2.0868 - val_acc: 0.3754 
# 131  199ms/step - loss: 0.7650 - acc: 0.7808 - val_loss: 0.6818 - val_acc: 0.8034
# 13    27ms/step - loss: 0.6413 - acc: 0.8194 - val_loss: 0.5457 - val_acc: 0.8517 1-LSTM
# 13    30ms/step - loss: 0.6472 - acc: 0.8171 - val_loss: 0.5493 - val_acc: 0.8717 2-LSTM 
# 13    36ms/step - loss: 0.5443 - acc: 0.8493 - val_loss: 0.4665 - val_acc: 0.8766 3-LSTM 
# 1     20ms/step - loss: 0.7527 - acc: 0.7846 - val_loss: 0.6238 - val_acc: 0.8288

# MLP
# 131  199ms/step - loss: 4.6698 - acc: 0.3628 - val_loss: 2.0174 - val_acc: 0.4895
# 13  

# CNN
# 13  27ms/step - loss: 1.9294 - acc: 0.4842 - val_loss: 1.9757 - val_acc: 0.5213


# In[206]:


pgi = PredictIntent(is_general = True)
pgi.batch_size = 13
pgi.max_sequence_length = 5
pgi.intent_embedding_dim = 10
pgi.num_units = 30
pgi.validation_split = 0.2
pgi.random_state = 42
pgi.data_path = "../feature_and_vector_seq"


# In[207]:


pgi.build_MLP_model()
pgi_MLP = pgi.fit_generator(epochs = 100)
save_model(K.get_session(), pgi_MLP, 'pgi_MLP.pb')


# In[196]:


pgi.build_CNN_model()
pgi_CNN = pgi.fit_generator(epochs = 100)
save_model(K.get_session(), pgi_CNN, 'pgi_CNN.pb')


# In[198]:


pgi.build_RNN_model()
pgi_RNN = pgi.fit_generator(epochs = 100)
save_model(K.get_session(), pgi_RNN, 'pgi_RNN.pb')


# In[199]:


pgi.build_BiRNN_model()
pgi_BiRNN = pgi.fit_generator(epochs = 100)
save_model(K.get_session(), pgi_BiRNN, 'pgi_BiRNN.pb')

