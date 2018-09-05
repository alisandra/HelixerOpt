__author__ = 'Alisandra Denton'

import numpy as np
from .sequence import fasta2seqs, atcg2numbers, reverse_complement




def row_to_array(row, dtype=np.float32):
    array = np.array(row)
    array = array.astype(dtype)
    return array


def seq_to_row(seq, spread_evenly=True):
    return atcg2numbers(seq, spread_evenly=spread_evenly)




def padded_subseq(sequence, start, end):
    # pad at start
    pfx = ''
    if start < 0:
        pfx = (start * -1) * 'N'
        start = 0

    # pad at end
    sfx = ''
    l = len(sequence)
    if end > l:
        sfx = (end - l) * 'N'
        end = l

    return pfx + sequence[start:end] + sfx




def chunkstring(string, length):
    return (string[0+i:length+i] for i in range(0, len(string), length))


def write_seq(file_handle, seq, seqid, new_length=60):
    file_handle.write('>' + seqid + '\n')
    for substring in chunkstring(seq, new_length):
        file_handle.write(substring + '\n')
