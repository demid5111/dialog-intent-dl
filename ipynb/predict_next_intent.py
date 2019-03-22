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

