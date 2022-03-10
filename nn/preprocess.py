# BMI 203 Project 7: Neural Network


# Importing Dependencies
import numpy as np
from typing import List, Tuple
from numpy.typing import ArrayLike

# Defining preprocessing functions
def one_hot_encode_seqs(seq_arr: List[str]) -> ArrayLike:
    """
    This function generates a flattened one hot encoding of a list of nucleic acid sequences
    for use as input into a fully connected neural net.

    Args:
        seq_arr: List[str]
            List of sequences to encode.

    Returns:
        encodings: ArrayLike
            Array of encoded sequences, with each encoding 4x as long as the input sequence
            length due to the one hot encoding scheme for nucleic acids.

            For example, if we encode 
                A -> [1, 0, 0, 0]
                T -> [0, 1, 0, 0]
                C -> [0, 0, 1, 0]
                G -> [0, 0, 0, 1]
            Then, AGA -> [1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]
    """
    encodings = []
    for seq in seq_arr:
    	e = np.array([])
    	for n in seq:
    		if n == 'A':
    			e = np.append(e, [1, 0, 0, 0])
    		elif n == 'T':
    			e = np.append(e, [0, 1, 0, 0])
    		elif n == 'C':
    			e = np.append(e, [0, 0 ,1, 0])
    		elif n == 'G':
    			e = np.append(e, [0, 0, 0, 1])
    		else:
    			raise ValueError('Invalid nucleotide in sequence')
    	encodings.append(e)
    return(encodings)


def sample_seqs(
        seqs: List[str],
        labels: List[bool]) -> Tuple[List[str], List[bool]]:
    """
    This function should sample your sequences to account for class imbalance. 
    Consider this as a sampling scheme with replacement.
    
    Args:
        seqs: List[str]
            List of all sequences.
        labels: List[bool]
            List of positive/negative labels

    Returns:
        sampled_seqs: List[str]
            List of sampled sequences which reflect a balanced class size
        sampled_labels: List[bool]
            List of labels for the sampled sequences
    """    
    # get minority class -- if the sum of the labels (T = 1, F = 0) is less than
    # half the length of the label list then there are more Falses than Trues
    min_class = sum(labels) < (len(labels)/2)
    i_class_min = np.where(labels == min_class)[0]
    i_class_max = np.where(labels != min_class)[0]
    n_class_min = len(i_class_min)
    n_class_max = len(i_class_max)
    print("number of samples in minority class: " + str(n_class_min))
    print("number of samples in majority class: " + str(n_class_max))
    
    # upsample from minority class to match size of majority class
    upsample_class_min = np.random.choice(i_class_min, size = n_class_max, replace = True)
    sampled_seqs = np.hstack((seqs[upsample_class_min], seqs[i_class_max]))
    sampled_labels = np.hstack((labels[upsample_class_min], labels[i_class_max]))
    return(sampled_seqs, sampled_labels)

def trim_seqs(seqs: List[str], n: int) -> List[str]:
	"""
	This function trims your sequences to length n to account for
	differences in sequence length.
	
	Args:
		seqs: List[str]
			List of sequences
		n: int
			Target length to trim to
	
	Return:
		trimmed_seqs: List[str]
			List of sequences trimmed to length n
	"""
	trimmed_seqs = []
	for seq in seqs:
		l = len(seq)
		i = np.random.choice(l - n)
		new_seq = seq[i:i+n]
		trimmed_seqs.append(new_seq)
	return(trimmed_seqs)
