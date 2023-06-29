import Bio.SeqIO as SeqIO
import pickle
from Bio import motifs
import sys
def get_motifs_list(motifs_file):

    """
    Returns array containing Bio.motifs.jaspar.Motif objects
    """
    with open(motifs_file, "r") as jspr:
        return motifs.parse(jspr, "jaspar")

def get_position_vector(sequence, jaspar_motifs):
    """
    Returns dictionary of motif_name: [positions] for each motif in jaspar_motifs
    Parameters:
    sequence: Bio.Seq object
    jaspar_motifs: array of Bio.motifs.jaspar.Motif objects

    Returns:
    motif_positions: dictionary of motif_name: [positions] for each motif in jaspar_motifs
    """
    motif_positions = {motif.name: [] for motif in jaspar_motifs}
    for motif in jaspar_motifs:
        for pos, seq in motif.pssm.search(sequence, threshold=1, both=False):
            motif_positions[motif.name].append(pos)
    return motif_positions

def position_dict_to_occurence(positions_dict):
    """
    Returns dictionary of motif_name: occurence for each motif in positions_dict
    Parameters:
    positions_dict: dictionary of motif_name: [positions] for each motif in jaspar_motifs
    
    out_list = []
    for motif_name in positions_dict:
        if len(positions_dict[motif_name]) > 0:
            out_list.append(motif_name)
    return out_list

seq_num = sys.argv[1]
jaspar_file = "/storage/brno2/home/xhorvat9/ltr-annotator/Diplomovka_Final/JASPAR_work/All_Profiles_JASPAR.jaspar"
motifs = get_motifs_list(jaspar_file)
seq_file = f"/storage/brno2/home/xhorvat9/ltr-annotator/Diplomovka_Final/1_Database_build/Negative_train_sequences/Negative_sequence_generation/all_length_non_LTRs_withMarkov{seq_num}.fasta"
seq_motif_list = {}
count = 0
for rec in SeqIO.parse(seq_file, "fasta"):
    if count % 1000 == 0:
        print(count)
    seq_motif_list[rec.id] = get_position_vector(rec.seq, motifs)
    count += 1
pickle.dump(seq_motif_list, open(f"/storage/brno2/home/xhorvat9/ltr-annotator/Diplomovka_Final/JASPAR_work/non_LTR_sequence_motifs{seq_num}.b", "wb+"))
