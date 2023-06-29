import tensorflow as tf
import numpy as np
import sys
import Bio.SeqIO as SeqIO
class NNPredictor():
    """
    Class that predicts LTR sequences using a neural network
    Parameters
    ----------
    short_model_path : str
        Path to the short sequence model
    medium_model_path : str
        Path to the medium sequence model
    long_model_path : str
        Path to the long sequence model
    """

    def __init__(self, short_model_path="additional_files/Trained_NNs/short_seq_model/", medium_model_path="additional_files/Trained_NNs/medium_seq_model/", long_model_path="additional_files/Trained_NNs/long_seq_model/"):
        self.models = {350: tf.keras.models.load_model(short_model_path), 
                       700: tf.keras.models.load_model(medium_model_path),
                       2000: tf.keras.models.load_model(long_model_path)}
    @staticmethod
    def onehote(seq):
        """
        Function that takes in a sequence and returns a one-hot encoded array
        Parameters
        ----------
        seq : str
            DNA sequence
        Returns
        -------
        np.array
            One-hot encoded array
        """
        seq2=list()
        mapping = {"A":[1., 0., 0., 0.], "C": [0., 1., 0., 0.], "G": [1., 0., 0., 0.], "T":[0., 0., 0., 1.], "N":[0., 0., 0., 0.]}
        for i in seq:
            seq2.append(mapping[i]  if i in mapping.keys() else [0., 0., 0., 0.]) 
        return np.array(seq2)


    def get_predictions(self, sequences, max_len):
        print(max_len)
        oneHotfeatures = [NNPredictor.onehote(s) for s in sequences]
        paddedDNA = tf.keras.preprocessing.sequence.pad_sequences(oneHotfeatures,padding="pre", maxlen=max_len)
        return ((self.models[max_len].predict(paddedDNA) > 0.5)+0).ravel()

    def predict(self, sequences):
        """
        Function that takes in a list of sequences and returns a list of predictions
        Parameters
        ----------
        sequences : list
            List of sequences
        Returns
        -------
        list
            List of predictions
        """
        seq_buckets = {350: [], 700: [], 2000: []}
        seq_indices = {350: [], 700: [], 2000: []}
        index = 0
        for seq in sequences:
            seq_len = len(seq)
            if seq_len <= 350:
                seq_buckets[350].append(seq)
                seq_indices[350].append(index)
            elif seq_len <= 700:
                seq_buckets[700].append(seq)
                seq_indices[700].append(index)
            else:
                seq_buckets[2000].append(seq)
                seq_indices[2000].append(index)
            index += 1
            
        predictions = []
        for b in seq_buckets:
            if len(seq_buckets[b]) == 0:
                predictions.append([])
                continue
            predictions.append(self.get_predictions(seq_buckets[b], b))
            
        preds = np.zeros(len(sequences))
        for b, k in zip(seq_indices, range(3)):
            for i in range(len(seq_indices[b])):
                preds[seq_indices[b][i]] = predictions[k][i]        
        return preds
    
if __name__ == '__main__':
    NNP = NNPredictor()
    predictions = NNP.predict([str(rec.seq) for rec in SeqIO.parse(sys.argv[1], "fasta")])
    np.savetxt("predictions.np", predictions)