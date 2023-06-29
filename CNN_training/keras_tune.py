import wandb
from wandb.keras import WandbCallback
import keras_tuner as kt
import numpy as np
import Bio.SeqIO as SeqIO
import keras
from sklearn.model_selection import train_test_split
import tensorflow as tf

def onehote(seq):
    """
    One Hot encoding function
    """
    seq2=list()
    mapping = {"A":[1., 0., 0., 0.], "C": [0., 1., 0., 0.], "G": [1., 0., 0., 0.], "T":[0., 0., 0., 1.], "N":[0., 0., 0., 0.]}
    for i in seq:
      seq2.append(mapping[i]  if i in mapping.keys() else [0., 0., 0., 0.]) 
    return np.array(seq2)

class MyTuner(kt.Tuner):
  """
    Custom Tuner class that inherits from keras_tuner.Tuner
  """
  def run_trial(self, trial, trainX, trainY, batch_size, epochs, objective, arch_type):
      hp = trial.hyperparameters
      objective_name_str = objective

      ## create the model with the current trial hyperparameters
      model = self.hypermodel.build(hp)

      ## Initiates new run for each trial on the dashboard of Weights & Biases
      run = wandb.init(project="Testing700+", config=hp.values, name=arch_type)

      ## WandbCallback() logs all the metric data such as
      ## loss, accuracy and etc on dashboard for visualization
      history = model.fit(trainX,
                trainY,
                batch_size=batch_size,
                epochs=epochs,
                validation_split=0.1,
                callbacks=[WandbCallback()])  


      ## if val_accurcy used, use the val_accuracy of last epoch model which is fully trained
      val_acc = history.history['val_binary_accuracy'][-1] 

      ## Send the objective data to the oracle for comparison of hyperparameters
      self.oracle.update_trial(trial.trial_id, {objective_name_str:val_acc})

      ## ends the run on the Weights & Biases dashboard
      run.finish()


import random 
max_len = 2500
sequences = [str(rec.seq) for rec in SeqIO.parse("/content/drive/MyDrive/sequences/LTRs_700+.fasta", "fasta")]
negative_sequences = [str(rec.seq) for rec in SeqIO.parse("/content/drive/MyDrive/sequences/non_LTRs_700+.fasta", "fasta")]

# truncate sequences longer than 2000
def trunc(sequence, length):
    """
    Truncate sequences longer than length at both ends
    """
    if len(sequence) > max_len:
        trunc_size = (len(sequence)-length)//2 + 1
        return sequence[trunc_size:len(sequence)-trunc_size]
    else:
        return sequence

# truncate sequences
sequences = [trunc(seq, max_len) for seq in sequences]
negative_sequences = [trunc(seq, max_len) for seq in negative_sequences]


# transform and pad sequences
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

# Split the data into training and testing sets
trainX, testX, trainY, testY = train_test_split(encodedDNA, labels, test_size=0.2, random_state=42)
wandb.init(project="ComparingLayers700+", name="2CNN_1LSTM")

def build_large_model(hp):
    """
    Sets up the interface for keras tuner with the hyperparameters to be tuned
    """
    model = keras.Sequential()
    model.add(keras.layers.Conv1D(
        filters = hp.Choice('filters', [ 32, 64, 128, 200]),
        kernel_size = hp.Choice('kernel_size', [5, 15, 25, 35]), 
        input_shape=trainX[0].shape, 
        padding="same"))
  
    model.add(keras.layers.MaxPooling1D(
        pool_size = hp.Choice("pool_size", [2, 4])))
    
    model.add(keras.layers.Conv1D(
        filters = hp.Choice('filters2', [ 32, 64, 128, 200]),
        kernel_size = hp.Choice('kernel_size2', [5, 15, 25, 35]), 
        padding="same"))
    
    model.add(keras.layers.MaxPooling1D(
        pool_size = hp.Choice("pool_size2", [2, 4])))
    
    model.add(keras.layers.Conv1D(
        filters = hp.Choice('filters3', [ 32, 64, 128, 200]),
        kernel_size = hp.Choice('kernel_size3', [5, 15, 25, 35]), 
        padding="same"))
    
    model.add(keras.layers.MaxPooling1D(
        pool_size = hp.Choice("pool_size3", [2, 4])))
    
    # Add an LSTM layer
    model.add(keras.layers.LSTM(
        units = hp.Choice("lstm_units", [64, 128])))

    model.add(keras.layers.Dense(
        hp.Choice('units', [8, 32, 64]),
        activation='relu'))
  
    model.add(keras.layers.Dense(
        units = 1,
        activation='sigmoid'))
    
    model.compile(loss='binary_crossentropy', optimizer="adam", metrics=['binary_accuracy'])
    return model

# create the tuner object
tuner1 = MyTuner(
    oracle=kt.oracles.RandomSearchOracle(
        objective=kt.Objective('val_binary_accuracy', 'max'),
        max_trials=25),
    hypermodel=build_large_model,
    directory='results',
    project_name='0_350_3CNN1LSTM')
# run the search
tuner1.search(trainX, trainY, 500, 25, "val_binary_accuracy", "3CNN1LSTM")