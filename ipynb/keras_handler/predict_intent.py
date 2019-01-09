import os
import tensorflow as tf
import keras
from keras.layers import Dense, Input, GlobalMaxPooling1D, Flatten, TimeDistributed, Bidirectional
from keras.layers import Conv1D, MaxPooling1D, Embedding, LSTM, Dropout, Activation, Permute
from keras import regularizers
from keras.models import Model, Sequential
from keras.callbacks import TensorBoard
from keras.backend import permute_dimensions
from keras_handler.sequence_generator import SequenceGenerator

class PredictIntent():
    intents = {"": 0, " ": 0, "а": 1, "a": 1, "б": 2, "в": 3, "г": 4, "д": 5,
               "е": 6, "e": 6, "ж": 7, "з": 8, "3": 8, "и": 9, "к": 10,
               "л": 11, "м": 12, "н": 13, "о": 14, "п": 15,
               "р": 16, "с": 17, "т": 18, "у": 19, "ф": 20,
               "х": 21, "ц": 22, "ч": 23, "ш": 24, "щ": 25}
    general_intents = {"": 0, " ": 0, "а": 1, "a": 1, "б": 1, "в": 1, "г": 1, "д": 1,  # Информативно-воспроизводящий
                       "е": 2, "e": 2, "ж": 2, "з": 2, "3": 2, "и": 2, "к": 2,  # Эмотивно-консолидирующий
                       "л": 3, "м": 3, "н": 3, "о": 3, "п": 3,  # Манипулятивный тип, доминирование
                       "р": 4, "с": 4, "т": 4, "у": 4, "ф": 4,  # Волюнтивно-директивный
                       "х": 5, "ц": 5, "ч": 5, "ш": 5, "щ": 5}  # Контрольно-реактивный
    batch_size = 13
    max_sequence_length = 5
    intent_embedding_dim = 10
    num_units = 30
    validation_split = 0.2
    random_state = 42
    data_path = "../feature_and_vector_seq"
    tb = TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=32, write_graph=True, write_grads=False,
                     write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)

    def __init__(self, is_general=False):
        if is_general:
            self.intent_index = self.general_intents
        else:
            self.intent_index = self.intents
        self.num_intents = max(self.intent_index.values()) + 1
        self.sg = SequenceGenerator(self.data_path, self.intent_index, self.max_sequence_length, self.validation_split,
                                    only_last=False,
                                    random_state=self.random_state)
        print('num_intents', self.num_intents)

    def build_CNN_model(self):
        embedding_layer = Embedding(self.num_intents,
                                    self.intent_embedding_dim,
                                    input_length=self.max_sequence_length - 1,
                                    trainable=True)

        sequence_input = Input(shape=(self.max_sequence_length - 1,), dtype='int32')
        embedded_sequences = embedding_layer(sequence_input)
        #         x_p = permute_dimensions(embedded_sequences, pattern=[0, 2, 1])

        x = Conv1D(128, 2, activation='relu', kernel_regularizer=regularizers.l2(0.0001))(embedded_sequences)
        x = MaxPooling1D(1)(x)
        x = Dropout(0.2)(x)
        x = Conv1D(128, 3, activation='relu', kernel_regularizer=regularizers.l2(0.0001))(x)
        x = GlobalMaxPooling1D()(x)
        x = Dropout(0.2)(x)
        x = Dense(128, activation='relu')(x)
        preds = Dense(self.num_intents, activation='softmax')(x)

        model = Model(sequence_input, preds)
        model.compile(loss='categorical_crossentropy',
                      optimizer='rmsprop',
                      metrics=['acc'])
        self.model = model
        self.sg.only_last = True
        return self.model

    def build_MLP_model(self):
        embedding_layer = Embedding(self.num_intents,
                                    self.intent_embedding_dim,
                                    input_length=self.max_sequence_length - 1,
                                    trainable=True)

        model = Sequential()
        model.add(embedding_layer)
        model.add(Flatten())
        model.add(Dense(128, activation='relu', name="Dense1"
                        #                         , activity_regularizer=regularizers.l1(0.009)
                        , kernel_regularizer=regularizers.l2(0.0001)
                        #                         , bias_regularizer = regularizers.l2(0.009)
                        ))  #
        #         model.add(Dropout(0.2))
        model.add(Dense(self.num_intents, activation='softmax', name="Dense2"
                        #                         , activity_regularizer=regularizers.l1(0.009)
                        , kernel_regularizer=regularizers.l2(0.0001)
                        #                         , bias_regularizer = regularizers.l2(0.009)
                        ))
        #         model.add(Dropout(0.2))
        model.compile(loss='categorical_crossentropy',
                      optimizer='rmsprop',
                      metrics=['acc'])
        self.model = model
        self.sg.only_last = True
        return self.model

    def build_RNN_model(self):
        embedding_layer = Embedding(self.num_intents,
                                    self.intent_embedding_dim,
                                    input_length=self.max_sequence_length - 1,
                                    trainable=True)

        sequence_input = Input(shape=(self.max_sequence_length - 1,), dtype='int32')
        embedded_sequences = embedding_layer(sequence_input)

        #         x_t = permute_dimensions(embedded_sequences, pattern=[0, 2, 1])
        #         print('embedded_sequences',embedded_sequences.shape)
        #         print('x_t',x_t.shape)

        x = LSTM(self.num_units, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)(embedded_sequences)
        x = LSTM(self.num_units, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)(x)
        preds = TimeDistributed(Dense(self.num_intents, activation='softmax'))(x)
        #         preds = Dense(self.num_intents, activation='softmax')(x)
        #         print('preds.shape',preds.shape)

        model = Model(sequence_input, preds)
        model.compile(loss='categorical_crossentropy',
                      optimizer='rmsprop',
                      metrics=['acc'])
        self.model = model
        self.sg.only_last = False
        return self.model

    def build_BiRNN_model(self):
        embedding_layer = Embedding(self.num_intents,
                                    self.intent_embedding_dim,
                                    input_length=self.max_sequence_length - 1,
                                    trainable=True)
        model = Sequential()
        model.add(embedding_layer)
        #         model.add(Permute([1, 2]))
        model.add(Bidirectional(LSTM(self.num_units, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)))
        model.add(Bidirectional(LSTM(self.num_units, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)))
        #         model.add(Bidirectional(LSTM(self.num_units, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)))
        model.add(
            TimeDistributed(Dense(self.num_intents, activation='softmax', name="Dense2"), name="TimeDistributed2"))
        model.compile(loss='categorical_crossentropy',
                      optimizer='rmsprop',
                      metrics=['acc'])
        self.model = model
        return self.model

    def RNN_attention():

        embedding_layer = Embedding(self.num_intents,
                                    self.intent_embedding_dim,
                                    input_length=self.max_sequence_length - 1,
                                    trainable=True)
        model1 = Sequential()
        model1.add(embedding_layer)
        model1.add(LSTM(self.num_units, return_sequences=True))

        model2 = Sequential()
        model2.add(Dense(input_dim=input_dim, output_dim=step))
        model2.add(Activation('softmax'))  # Learn a probability distribution over each  step.
        # Reshape to match LSTM's output shape, so that we can do element-wise multiplication.
        model2.add(RepeatVector(self.num_units))
        model2.add(Permute(2, 1))

        model = Sequential()
        model.add(
            Merge([model1, model2], 'mul'))  # Multiply each element with corresponding weight a[i][j][k] * b[i][j]
        model.add(TimeDistributedMerge('sum'))  # Sum the weighted elements.

        attention = Dense(1, activation='tanh')(activations)
        attention = Flatten()(attention)
        attention = Activation('softmax')(attention)
        attention = RepeatVector(units)(attention)
        attention = Permute([2, 1])(attention)

        sent_representation = merge([activations, attention], mode='mul')
        model.compile(loss='categorical_crossentropy',
                      optimizer='rmsprop',
                      metrics=['acc'])
        self.model = model
        self.sg = SequenceGenerator(self.data_path, self.intent_index, self.max_sequence_length, self.validation_split,
                                    only_last=False)
        return self.model

    def fit_generator(self, epochs):
        history_nn = self.model.fit_generator(
            generator=self.sg.generate_batch(self.batch_size, subset='training'),
            steps_per_epoch=len(os.listdir(self.data_path)) * (1 - self.validation_split) // self.batch_size,
            epochs=epochs,
            validation_data=self.sg.generate_batch(self.batch_size, subset='validation'),
            validation_steps=len(os.listdir(self.data_path)) * self.validation_split // self.batch_size + 1,
            callbacks=[self.tb]
        )
        return history_nn