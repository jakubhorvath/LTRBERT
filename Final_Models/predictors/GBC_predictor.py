import pickle
import tqdm
from sklearn.base import TransformerMixin
import pandas as pd
import Bio.SeqIO as SeqIO 
import sys
from Bio import motifs
import numpy as np
class JASPARParser(TransformerMixin):
    """
    Class that parses JASPAR motifs and returns a pandas DataFrame of motif counts in a sequence
    Parameters
    ----------
    jaspar_motif_file : str
        Path to JASPAR motif file
    Returns
    -------
    pandas.DataFrame
        DataFrame of motif counts
    """
    def __init__(self, jaspar_motif_file="additional_files/All_Profiles_JASPAR.jaspar") -> None:
        #load the motifs
        self.motifs = {}
        with open(jaspar_motif_file, "r") as jspr:
            self.motifs = motifs.parse(jspr, "jaspar")

    def get_position_vector(self, sequence):
        # for the sequence returns a vector of positions of all motifs
        motif_positions = {motif.name: [] for motif in self.motifs}
        for motif in self.motifs:
            for pos, seq in motif.pssm.search(sequence, threshold=1, both=False):
                motif_positions[motif.name].append(pos)
        return motif_positions

    def get_motif_counts(self, motif_dict_count: dict, TF_sites: dict):
        for seq in TF_sites:
            for motif in motif_dict_count:
                if len(TF_sites[seq][motif]) > 0:
                    motif_dict_count[motif].append(len(TF_sites[seq][motif]))
                else:
                    motif_dict_count[motif].append(0)

    def fit_transform(self, X, y = None, **fit_params) -> list:
        seq_motif_list = {}
        id = 0
        for seq in tqdm.tqdm(X):
            seq_motif_list[str(id)] = self.get_position_vector(seq)
            id += 1
        motif_count = dict([(key, []) for key in seq_motif_list[list(seq_motif_list.keys())[0]]])
        self.get_motif_counts(motif_count, seq_motif_list)
        return pd.DataFrame(motif_count)

    def transform(self, X, **transform_params):
        return self.fit_transform(X, **transform_params)

if __name__ == '__main__':

    TFIDF_GBC_pipeline = pickle.load(open("./additional_files/TFIDF_GBC_pipeline.b", "rb"))
    JP = JASPARParser("./additional_files/All_Profiles_JASPAR.jaspar")
    if len(sys.argv) == 2 or sys.argv[2] == "classes":
        predictions = TFIDF_GBC_pipeline.predict(JP.fit_transform([str(rec.seq) for rec in SeqIO.parse(sys.argv[1], "fasta")]))
    elif sys.argv[2] == "probabilities":
        predictions = TFIDF_GBC_pipeline.predict_proba(JP.fit_transform([str(rec.seq) for rec in SeqIO.parse(sys.argv[1], "fasta")]))
    else:
        print("Unrecognized argument:", sys.argv[2], "Please provide \'classes\' for class output or \'probabilities\' to get the prediction probabilites")
        sys.exit(1)
    np.savetxt("./predictions.np", predictions)