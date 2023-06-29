import keras
import keras_tuner as kt
import tensorflow as tf
import wandb
from sklearn.model_selection import train_test_split
import Bio.SeqIO as SeqIO
import random
import numpy as np
from wandb.keras import WandbCallback
import sys

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def build_model1():
    """
    1 CNN layer, 1 LSTM
    """
    model = keras.Sequential()

    # First convolutional layer with 32 filters and a kernel size of 8
    model.add(keras.layers.Conv1D(32, 8, activation='relu', input_shape=trainX[0].shape))
    # Pooling layer with pool size of 4
    model.add(keras.layers.MaxPooling1D(4))
    # Second convolutional layer with 64 filters and a kernel size of 8
    model.add(keras.layers.Conv1D(64, 8, activation='relu'))
    # Pooling layer with pool size of 4
    model.add(keras.layers.MaxPooling1D(4))
    # Third convolutional layer with 128 filters and a kernel size of 8
    model.add(keras.layers.Conv1D(128, 8, activation='relu'))
    # LSTM layer with 64 units
    model.add(keras.layers.LSTM(64))
    # Dropout layer to prevent overfitting
    model.add(keras.layers.Dropout(0.5))
    # Dense layer with 64 units and relu activation
    model.add(keras.layers.Dense(64, activation='relu'))
    # Output layer with softmax activation for multi-class classification
    model.add(keras.layers.Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer="adam", metrics=['binary_accuracy'])
    return model

def build_model2():
    """
    2 CNN layers, 1 LSTM
    """
    model = keras.Sequential()
    model.add(keras.layers.Conv1D(
        filters = 64,
        kernel_size = 5, 
        input_shape=trainX[0].shape, 
        padding="same"))
  
    model.add(keras.layers.MaxPooling1D(
        pool_size = 2))

    model.add(keras.layers.Conv1D(
        filters = 64,
        kernel_size = 15, 
        input_shape=trainX[0].shape, 
        padding="same"))
  
    model.add(keras.layers.MaxPooling1D(
        pool_size = 2))
    

    # Add an LSTM layer
    model.add(keras.layers.LSTM(
        units = 128))

    model.add(keras.layers.Dense(
        64,
        activation='relu'))
  
    model.add(keras.layers.Dense(
        units = 1,
        activation='sigmoid'))
    
    model.compile(loss='binary_crossentropy', optimizer="adam", metrics=['binary_accuracy'])
    return model

def build_model3():

    model = keras.Sequential()
    model.add(keras.layers.Conv1D(filters=320, kernel_size=8, activation='relu', input_shape=trainX[0].shape))
    model.add(keras.layers.MaxPooling1D(pool_size=4))
    model.add(keras.layers.Conv1D(filters=480, kernel_size=8, activation='relu'))
    model.add(keras.layers.MaxPooling1D(pool_size=4))
    model.add(keras.layers.Conv1D(filters=960, kernel_size=8, activation='relu'))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(units=925, activation='relu'))
    model.add(keras.layers.Dense(units=1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])
    return model


def build_model4():

    model = keras.Sequential()

    # First convolutional layer with 32 filters and a kernel size of 8
    model.add(keras.layers.Conv1D(32, 8, activation='relu', input_shape=trainX[0].shape))
    # Pooling layer with pool size of 4
    model.add(keras.layers.MaxPooling1D(4))
    # Second convolutional layer with 64 filters and a kernel size of 8
    model.add(keras.layers.Conv1D(64, 8, activation='relu'))
    # Pooling layer with pool size of 4
    model.add(keras.layers.MaxPooling1D(4))
    # Third convolutional layer with 128 filters and a kernel size of 8
    model.add(keras.layers.Conv1D(128, 8, activation='relu'))
    # Flatten layer to convert the output to a 1D feature vector
    model.add(keras.layers.Flatten())
    # Dropout layer to prevent overfitting
    model.add(keras.layers.Dropout(0.5))
    # Dense layer with 64 units and relu activation
    model.add(keras.layers.Dense(64, activation='relu'))
    # Output layer with softmax activation for multi-class classification
    model.add(keras.layers.Dense(1, activation='sigmoid'))

    # Compile the model
    model.compile(loss='binary_crossentropy',
                optimizer='adam',
                metrics=['binary_accuracy'])


def onehote(seq):
    """
    One Hot encoding function
    """
    seq2=list()
    mapping = {"A":[1., 0., 0., 0.], "C": [0., 1., 0., 0.], "G": [1., 0., 0., 0.], "T":[0., 0., 0., 1.], "N":[0., 0., 0., 0.]}
    for i in seq:
      seq2.append(mapping[i]  if i in mapping.keys() else [0., 0., 0., 0.]) 
    return np.array(seq2)


max_len = 2000

sequences = [str(rec.seq) for rec in SeqIO.parse("/var/tmp/xhorvat9_dip/training_DNABERT/sequences/LTRs_700+.fasta", "fasta")]
#sequences += [str(rec.seq.complement()) for rec in SeqIO.parse(positive_seq_file, "fasta")]
negative_sequences = [str(rec.seq) for rec in SeqIO.parse("/var/tmp/xhorvat9_dip/training_DNABERT/sequences/non_LTRs_700+.fasta", "fasta")]
#negative_sequences += [str(rec.seq.complement()) for rec in SeqIO.parse(negative_seq_file, "fasta")]

# truncate sequences longer than 2000
def trunc(sequence, length):
    if len(sequence) > max_len:
        trunc_size = (len(sequence)-length)//2 + 1
        return sequence[trunc_size:len(sequence)-trunc_size]
    else:
        return sequence

sequences = [trunc(seq, max_len) for seq in sequences]
negative_sequences = [trunc(seq, max_len) for seq in negative_sequences]


oneHotfeatures = []
for se in sequences:
    oneHotfeatures.append(onehote(se.upper()))
paddingDNA = tf.keras.preprocessing.sequence.pad_sequences(oneHotfeatures,padding="pre", maxlen=max_len)

sequence_lens = [len(sequence) for sequence in sequences]
random.seed(a=42, version=2)    

negative_oneHot = [onehote(s) for s in negative_sequences]
paddingNegativeDNA = tf.keras.preprocessing.sequence.pad_sequences(negative_oneHot, padding="pre", maxlen=max_len)

labels = np.array([1] * len(sequences) + [0] * len(negative_sequences))
encodedDNA = np.concatenate((paddingDNA, paddingNegativeDNA))

trainX, testX, trainY, testY = train_test_split(encodedDNA, labels, test_size=0.2, random_state=42)

CNN1_LSTM1 = build_model1()
CNN2_LSTM1 = build_model2()
CNN3 = build_model3()
CNN3_small = build_model4()

if sys.argv[1] == "1":
    wandb.init(project="ComparingLayers700+", name="3CNN_large")

    CNN3.fit(trainX, trainY, epochs=15, validation_split=0.2, batch_size=60, verbose=2, callbacks=[WandbCallback()])
elif sys.argv[1] == "2":
    wandb.init(project="ComparingLayers700+", name="3CNN_small")

    CNN3.fit(trainX, trainY, epochs=15, validation_split=0.2, batch_size=60, verbose=2, callbacks=[WandbCallback()])
elif sys.argv[1] == "3":
    wandb.init(project="ComparingLayers700+", name="3CNN_1LSTM")

    CNN1_LSTM1.fit(trainX, trainY, epochs=15, validation_split=0.2, batch_size=60, verbose=2, callbacks=[WandbCallback()])
elif sys.argv[1] == "4":
    wandb.init(project="ComparingLayers700+", name="2CNN_1LSTM")

    CNN2_LSTM1.fit(trainX, trainY, epochs=15, validation_split=0.2, batch_size=60, verbose=2, callbacks=[WandbCallback()])

else:
    print("Invalid argument")
