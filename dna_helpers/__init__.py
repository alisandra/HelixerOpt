__author__ = 'Alisandra Denton'

import numpy as np


def atcg2numbers(seq, spread_evenly=True):
    # CATG, for easier reverse_complement
    decode = {'c': [1, 0, 0, 0],
              'a': [0, 1, 0, 0],
              't': [0, 0, 1, 0],
              'g': [0, 0, 0, 1],
              'y': [0.5, 0, 0.5, 0],
              'r': [0, 0.5, 0, 0.5],
              'w': [0, 0.5, 0.5, 0],
              's': [0.5, 0, 0, 0.5],
              'k': [0, 0, 0.5, 0.5],
              'm': [0.5, 0.5, 0, 0],
              'd': [0, 0.33, 0.33, 0.33],
              'v': [0.33, 0.33, 0, 0.33],
              'h': [0.33, 0.33, 0.33, 0],
              'b': [0.33, 0, 0.33, 0.33],
              'n': [0.25, 0.25, 0.25, 0.25]}
    # strip ambiguous codes to 0s if not spreading them evenly
    if not spread_evenly:
        for key in decode:
            if key not in ['a', 't', 'c', 'g']:
                decode[key] = [0, 0, 0, 0]

    onfail = [0, 0, 0, 0]
    seq = seq.lower()
    #out = []
    i = 0
    out = np.full((len(seq), 4), -1, dtype=np.float32)
    for letter in seq:
        try:
            out[i, :] = decode[letter]
        except KeyError:
            print('non DNA character: {0}, adding {1} for no match'.format(letter, onfail))
            out[i, :] = onfail
        i += 1
    return out


def row_to_array(row, dtype=np.float32):
    array = np.array(row)
    array = array.astype(dtype)
    return array


def seq_to_row(seq, spread_evenly=True):
    return atcg2numbers(seq, spread_evenly=spread_evenly)


def fasta2seqs(fastafile, headerchar=' '):
    seqs = {}
    running_seq = ''
    running_id = None
    with open(fastafile) as f:
        for line in f:
            line = line.rstrip()
            if line.startswith('>'):
                if len(running_seq) > 0:
                    seqs[running_id] = running_seq
                    running_seq = ''
                running_id = line.split(headerchar)[0][1:]
            else:
                running_seq += line
    # save last sequence too
    seqs[running_id] = running_seq
    return seqs


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


def reverse_complement(seq):
    fw = "ACGTMRWSYKVHDBN"
    rv = "TGCAKYWSRMBDHVN"
    fw += fw.lower()
    rv += rv.lower()
    key = {}
    for f, r in zip(fw, rv):
        key[f] = r
    rc_seq = ''
    for base in reversed(seq):
        try:
            rc_seq += key[base]
        except KeyError as e:
            raise KeyError('{} caused by non DNA character {}'.format(e, base))

    return rc_seq


def chunkstring(string, length):
    return (string[0+i:length+i] for i in range(0, len(string), length))


def write_seq(file_handle, seq, seqid, new_length=60):
    file_handle.write('>' + seqid + '\n')
    for substring in chunkstring(seq, new_length):
        file_handle.write(substring + '\n')
