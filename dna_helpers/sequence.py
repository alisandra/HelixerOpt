from __future__ import division
import numpy as np
import logging


class FastaReader(object):
    def __init__(self, fastafile, headerchar=' '):
        self.seqs = self._read_in(fastafile, headerchar)
        self.genome_length = self._genome_len()
        self.gc_fraction, self.kmer_content = self._gc_kmers()

    @staticmethod
    def _read_in(fastafile, headerchar=' '):
        return fasta2seqs(fastafile, headerchar)

    def _genome_len(self):
        out = 0
        for seq in self.seqs:
            out += len(seq)
        return out

    def _gc_kmers(self):
        base_counts = np.array([[0, 0, 0, 0]])
        kmer_counts = {}
        for seq in self.seqs:
            base_counts_new, kmer_counts_new = count_bases_kmers(seq)
            base_counts += base_counts_new  # keep cumulative counts of each base
            # keep cumulative counts of each cannonical 3mer
            for key in kmer_counts_new:
                add_or_init(kmer_counts, key, kmer_counts_new[key])
        gc = gc_percent(base_counts)
        return gc, kmer_counts


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


def count_bases_kmers(seq):
    counting_bases = np.array([[0., 0, 0, 0]])
    three_mer = ''
    kmer_counts = {}
    for bp in seq:
        counting_bases += atcg2numbers(bp)
        if len(three_mer) == 3:  # drop oldest base in 3mer
            three_mer = three_mer[1:]
        three_mer += bp  # add new base to 3mer
        if len(three_mer) <= 2:  # skip first 2bp at start before we have our 3-mer together
            pass
        else:
            cannonical = get_cannonical(three_mer)
            add_or_init(kmer_counts, cannonical)

    return counting_bases, kmer_counts


def gc_percent(bases_counts):
    gc_count = np.sum(bases_counts[:, IN_GC])
    full_count = np.sum(bases_counts)
    return gc_count / full_count


def add_or_init(a_dict, key, to_add=1):
    try:
        a_dict[key] += to_add
    except KeyError:
        a_dict[key] = to_add


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


def get_cannonical(kmer):
    # we're not messing with ambiguous kmers
    kmer = kmer.lower()
    rc = reverse_complement(kmer)

    if not all([x in ["a", "c", "g", "t"] for x in kmer]):
        return "n" * len(kmer)

    return sorted([kmer, rc])[0]


# ACGT for easy: Cannonical, ReverseComplement
ALL_EMBED = {'a': [1, 0, 0, 0],
             'c': [0, 1, 0, 0],
             'g': [0, 0, 1, 0],
             't': [0, 0, 0, 1],
             'r': [0.5, 0, 0.5, 0],
             'y': [0, 0.5, 0, 0.5],
             's': [0, 0.5, 0.5, 0],
             'w': [0.5, 0, 0, 0.5],
             'k': [0, 0, 0.5, 0.5],
             'm': [0.5, 0.5, 0, 0],
             'b': [0, 1/3, 1/3, 1/3],
             'h': [1/3, 1/3, 0, 1/3],
             'v': [1/3, 1/3, 1/3, 0],
             'd': [1/3, 0, 1/3, 1/3],
             'n': [0.25, 0.25, 0.25, 0.25]}


ACGT_EMBED = {'a': [1, 0, 0, 0],
              'c': [0, 1, 0, 0],
              'g': [0, 0, 1, 0],
              't': [0, 0, 0, 1]}

IN_GC = [False, True, True, False]

def atcg_sums(seq, spread_evenly=True):
    atcg_array = atcg2numbers(seq, spread_evenly)
    return np.sum(atcg_array, axis=0)


def atcg2numbers(seq, spread_evenly=True):
    # CATG, for easier reverse_complement
    if spread_evenly:
        decode = ALL_EMBED
    else:
        decode = ACGT_EMBED

    onfail = [0, 0, 0, 0]
    seq = seq.lower()
    #out = []
    i = 0
    out = np.full((len(seq), 4), -1, dtype=np.float32)
    for letter in seq:
        try:
            out[i, :] = decode[letter]
        except KeyError:
            logging.debug('non DNA character: {0}, adding {1} for no match'.format(letter, onfail))
            out[i, :] = onfail
        i += 1
    return out