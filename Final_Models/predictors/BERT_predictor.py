import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import tqdm
import sys
import Bio.SeqIO as SeqIO
import numpy as np
import tensorflow as tf

class Dataset(torch.utils.data.Dataset):
    """
    Class that creates a torch dataset object
    Parameters
    ----------
    encodings : dict
        Dictionary of encoded sequences
    labels : list
        List of labels
    Returns
    -------
    torch.utils.data.Dataset
        Torch dataset object
    """
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels:
            item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])


def Kmers_funct(seq, size=6):
    """
    Function that takes in a sequence and returns a list of kmers
    Parameters
    ----------
    seq : str
        DNA sequence
    size : int
        Size of kmers to return
    Returns
    -------
    list
        List of kmers
    """
    return [seq[x:x+size].upper() for x in range(len(seq) - size + 1)]

def tok_func(x): 
    """
    Function that takes in a sequence and returns a joined string of kmers
    Parameters
    ----------
    x : str
        DNA sequence
    Returns
    -------
    str
        Joined string of kmers
    """
    return " ".join(Kmers_funct(x))

class BERTPredictor():
    """
    Class that takes in a list of sequences and returns a list of predictions
    Parameters
    ----------
    pool_CNN_path : str
        Path to the pooled CNN model
    """
    def __init__(self, pool_CNN_path):
        self.tokenizer = BertTokenizer.from_pretrained('zhihan1996/DNA_bert_6')
        self.model350 = BertForSequenceClassification.from_pretrained('xhorvat9/LTR_BERT_0_350_noTSD', num_labels=2)
        self.model512 = BertForSequenceClassification.from_pretrained('xhorvat9/LTR_BERT_512_noTSD', num_labels=2)
        self.pool_CNN = tf.keras.models.load_model(pool_CNN_path)
        
        if torch.cuda.is_available():    
            device = torch.device("cuda")
        else:
            print('No GPU available, using the CPU instead.')
            device = torch.device("cpu")

        self.model350.to(device)
        self.model512.to(device)
        self.t350 = Trainer(self.model350)
        self.t512 = Trainer(self.model512)

    def get_pooled_predictions(self, seqs, return_prob):
        """ 
        Function that takes in a list of sequences above 700bps and returns a list of predictions 
            using the BERT-CNN pooling model
        Parameters
        ----------
        seqs : list
            List of sequences
        Returns
        -------
        list
            List of predictions
        """
        window_size = 350
        stride = 116 # ~ 1/3 of window size

        outputs = []
        sequences = []
        for seq in seqs:
            seq_windows = []
            for i in range(0, len(seq), stride):
                start = i
                end = i + window_size

                if end > len(seq):
                    end = len(seq)
                seq_windows.append(seq[start:end])
            sequences.append(seq_windows)

        counter = 0
        for s in tqdm.tqdm(sequences):
            if counter % 500 == 0 and counter != 0:
                print(f"processing sequence {counter}")
            tokenized = self.tokenizer([tok_func(x) for x in s], padding=True, truncation=True, max_length=350) # Create torch dataset
            test_dataset = Dataset(tokenized) # Load trained model
            embeddings, _,_ = self.t350.predict(test_dataset) # Preprocess raw predictions
            outputs.append(embeddings)
        padded_embeddings = tf.keras.preprocessing.sequence.pad_sequences(outputs, padding="pre", maxlen=45, dtype='float32')
        if return_prob:
            return self.pool_CNN.predict(padded_embeddings).ravel()
        else:
            return ((self.pool_CNN.predict(padded_embeddings) > 0.5)+0).ravel()
    def get_predictions(self, sequences, max_len, return_prob):
        """
        Function that takes in a list of sequences and returns a list of predictions
        Parameters
        ----------
        sequences : list
            List of sequences
        max_len : int
            Maximum length of sequences
        Returns
        -------
        list
            List of predictions
        """
        if max_len > 350:
            tokenized = self.tokenizer([tok_func(x) for x in sequences], padding=True, truncation=True, max_length=512)
            dt = Dataset(tokenized)
            raw_pred, _, _ = self.t512.predict(dt) # Preprocess raw predictions
        else:
            tokenized = self.tokenizer([tok_func(x) for x in sequences], padding=True, truncation=True, max_length=350)
            dt = Dataset(tokenized)
            raw_pred, _, _ = self.t350.predict(dt) # Preprocess raw predictions
        if return_prob:
            return raw_pred
        else:
            return np.argmax(raw_pred, axis=1)

    def predict(self, sequences, return_prob=False):
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
        seq_buckets = {350: [], 512: [], 700: []}
        seq_indices = {350: [], 512: [], 700: []}
        index = 0
        for seq in sequences:
            seq_len = len(seq)
            if seq_len <= 350:
                seq_buckets[350].append(seq)
                seq_indices[350].append(index)
            elif seq_len <= 700:
                seq_buckets[512].append(seq)
                seq_indices[512].append(index)
            else:
                seq_buckets[700].append(seq)
                seq_indices[700].append(index)
            index += 1

        out_predictions = []
        for b in seq_buckets:
            if len(seq_buckets[b]) == 0:
                out_predictions.append([])
                continue
            print(f"Processing sequences of length: <{b}")
            if b == 700:
                out_predictions.append(self.get_pooled_predictions(seq_buckets[b], return_prob))
            else:
                out_predictions.append(self.get_predictions(seq_buckets[b], b, return_prob))
        
        preds = np.zeros(len(sequences))
        for b, k in zip(seq_indices, range(3)):
            if len(seq_buckets[b]) == 0:
                continue
            for i in range(len(seq_indices[b])):
                preds[seq_indices[b][i]] = out_predictions[k][i]

        
        return preds

if __name__ == '__main__':
    BP = BERTPredictor("additional_files/TF_CNN_BERT_pool_model")
    predictions = BP.predict([str(rec.seq) for rec in SeqIO.parse(sys.argv[1], "fasta")], return_prob=False)
    np.savetxt("predictions.np", predictions)