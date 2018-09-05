from .sequence import count_bases_kmers, atcg_sums, get_cannonical, gc_percent
import numpy as np
import pytest


def count_gc(seq):
    base_counts, _ = count_bases_kmers(seq)
    return gc_percent(base_counts)


def test_gc():
    allgc = "GCGCGCGCGCCCCGGCGCGCGCGCccgcgccgcggggggg"
    assert count_gc(allgc) == 1
    halfgc = "CATGCATGcatgcatg"
    assert count_gc(halfgc) == 0.5
    ambig_half = "mmmmmkSWswCATGnnnn"
    assert count_gc(ambig_half) == 0.5


def test_amibiguity_codes():
    exact_ambig = (
        ('ATCG', 'NNNN'),
        ('CGT', 'BBB'),
        ('AGT', 'DDD'),
        ('ACT', 'HHH'),
        ('ACG', 'VVV'),
        ('GT', 'KK'),
        ('CT', 'YY'),
        ('CG', 'SS'),
        ('AT', 'WW'),
        ('AG', 'RR'),
        ('AC', 'MM')
    )
    for exact, ambig in exact_ambig:
        assert np.allclose(atcg_sums(exact), atcg_sums(ambig))


def test_cannonical():
    # make sure we get an error for clearly non-DNA input
    with pytest.raises(KeyError):
        get_cannonical('my bad')
    # make sure we get all ambiguous for ambiguous input
    assert get_cannonical("aabN") == "nnnn"
    # make sure a few successful cannonical selections work
    assert get_cannonical('atcg') == 'atcg'
    assert get_cannonical('aaaaA') == 'aaaaa'
    assert get_cannonical('tatat') == 'atata'
    assert get_cannonical('g') == 'c'


def test_threemer_count():
    _, kmer_count = count_bases_kmers('aaaaa')
    print(kmer_count)
    assert kmer_count['aaa'] == 3
    _, kmer_count = count_bases_kmers('atcga')
    assert kmer_count == {'atc': 1, 'cga': 2}
    _, kmer_count = count_bases_kmers('atatatata')
    assert kmer_count == {'ata': 7}