import os, random
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

class SequenceGenerator():
    def __init__(self, data_path, intent_index, max_sequence_length, validation_split, only_last=False,
                 random_state=None):
        self.data_path = data_path
        self.max_sequence_length = max_sequence_length
        self.intent_index = intent_index
        self.num_intents = max(intent_index.values()) + 1
        self.only_last = only_last
        self.file_list_train, self.file_list_test = self._split(validation_split, random_state=random_state)

    def _file2sequence(self, file_and_path):
        sequence = []
        for intent in pd.read_hdf(file_and_path, engine="python", encoding='cp1251')['Intent analysis']:
            if intent:
                intent_char = intent[0].lower()
            else:
                intent_char = ""
            sequence.append(self.intent_index[intent_char])
        return sequence

    def _split(self, validation_split, random_state=None):
        file_list = os.listdir(self.data_path)
        file_list_train, file_list_test = train_test_split(file_list, test_size=validation_split,
                                                           random_state=random_state)
        return file_list_train, file_list_test

    def __build_intent_sequence(self, dialogs_list):
        sequence_list = []
        for dialog in dialogs_list:
            sequence = []
            for intent in dialog['Intent analysis'].values:
                sequence.append(self.intent_index[intent])
            sequence_list.append(sequence)
        paded_sequences = pad_sequences(sequence_list, maxlen=self.max_sequence_length)
        return paded_sequences

    def generate_batch(self, batch_size, subset='training'):
        if subset == 'training':
            file_list = self.file_list_train
        elif subset == 'validation':
            file_list = self.file_list_test
        f_i = 0

        while True:
            i = 0
            sequence_batch = []
            while i < batch_size:
                if f_i == len(file_list):
                    f_i = 0
                    random.shuffle(file_list)
                file_and_path = os.path.join(self.data_path, file_list[f_i])
                sequence = self._file2sequence(file_and_path)
                if len(sequence) > self.max_sequence_length:
                    for ii in range(len(sequence) - self.max_sequence_length + 1):
                        sequence_i = sequence[ii:self.max_sequence_length + ii]
                        sequence_batch.append(sequence_i)
                        i += 1
                else:
                    sequence_i = sequence
                    sequence_batch.append(sequence_i)
                    i += 1
                f_i += 1
            paded_sequences = pad_sequences(sequence_batch, maxlen=self.max_sequence_length)
            inputs = paded_sequences[:, :-1]
            if self.only_last:
                labels = to_categorical(paded_sequences[:, -1:],
                                        num_classes=self.num_intents)  # for build_CNN_model build_MLP_model
            else:
                labels = to_categorical(paded_sequences[:, 1:],
                                        num_classes=self.num_intents)  # for build_BiRNN_model build_RNN_model
            yield inputs, labels

