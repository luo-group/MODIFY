import torch
from Bio import SeqIO


def softmax_mask(A, mask):
    """
    Softmax with mask

    Args:
        A: The phi matrix (num_position, 20)
        mask: The mask matrix (num_position, 20)

    Returns:
        q: The softmax matrix (num_position, 20)
    """

    A_max = torch.max(A,dim=1,keepdim=True)[0]
    A_exp = torch.exp(A-A_max)
    A_exp = A_exp * mask

    q = A_exp / A_exp.sum(dim=-1, keepdim=True)

    return q


def get_alphabet():
    """
    Get the alphabet of amino acids
    
    Returns:
        alphabet: The alphabet of amino acids
        map_a2i: The mapping from amino acid to index
        map_i2a: The mapping from index to amino acid
    """

    alphabet = 'ACDEFGHIKLMNPQRSTVWY'

    map_a2i = {j:i for i,j in enumerate(alphabet)}
    map_i2a = {i:j for i,j in enumerate(alphabet)}

    return alphabet, map_a2i, map_i2a


def load_sequence(path):
    """
    Load the sequence from a fasta file

    Args:
        path: The path to the fasta file

    Returns:
        sequence: The starting sequence
    """

    for record in SeqIO.parse(path, "fasta"):
        return str(record.seq)


def get_mask(positions, num_position, map_a2i, masked_AAs):
    """
    Get the mask matrix of a protein

    Args:
        protein: The protein name
        num_position: The number of positions to be mutated

    Returns:
        mask: The mask matrix (num_position, 20)
    """

    mask = torch.ones(num_position, len(map_a2i)).float()
    if masked_AAs != []:
        map_p2i = {j:i for i,j in enumerate(positions)}
        for masked_AA in masked_AAs:
            pos, aa = int(masked_AA[:-1]), masked_AA[-1]
            i = map_p2i[pos]
            a = map_a2i[aa]
            mask[i,a] = 0
    
    return mask

def init_worker_probability(d1, d2):
    global q, num_position
    q = d1
    num_position = d2

def calculate_probability(ind):
    ind = [(ind%20**(num_position-i))//20**(num_position-1-i) for i in range(num_position)]
    score = q[range(num_position),ind].prod().item()
    return score