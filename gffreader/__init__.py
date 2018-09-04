from __future__ import print_function
from __future__ import division

__author__ = 'Alisandra Denton & Janina Mass'

from builtins import range
import copy
import intervaltree
import pytest
import logging


class GFFLine(object):
    headers = ['seqid', 'source', 'type', 'start', 'end', 'score', 'strand', 'phase', 'attributes']

    def __repr__(self):
        return ",".join(self.ordered_list())

    def __init__(self, gffline=None, *initial_data, **kwargs):
        self.seqid = None
        self.source = None
        self.type = None
        self.start = None
        self.end = None
        self.score = None
        self.strand = None
        self.phase = None
        self.attributes = None
        self.comment = None
        self.region_type = MinimalData.no_info

        if not gffline.startswith('#'):
            gffline = gffline.rstrip()
            tmp = zip(GFFLine.headers, gffline.split("\t"))
        else:
            tmp = [('comment', gffline)]
        for key, val in tmp:
            if key in ['start', 'end', 'score', 'phase']:
                try:
                    val = int(val)
                except ValueError:
                    pass  # because score and phase often get some sort of '.' for NA/None value:573

            setattr(self, key, val)
        for dct in initial_data:
            for key in dct:
                setattr(self, key, dct[key])
        for key in kwargs:
            setattr(self, key, kwargs[key])
        self.children = []

    def gffstring(self):
        if self.seqid:
            ret = "\t".join(self.ordered_list())
        else:
            ret = self.comment
        return ret

    def ordered_list(self):
        out = [str(getattr(self, key)) for key in GFFLine.headers]
        return out

    def get_attribute(self, key, delimiter=';'):
        attrs = self.attributes
        search_key = key + '='
        p = None
        for a in attrs.split(delimiter):
            if a.startswith(search_key):
                p = a.split(search_key)[1]
        return p

    def change_attribute(self, key, value, delimiter=';'):
        attrs_list = self.attributes.split(delimiter)
        search_key = key + '='
        attrs_out = []
        for attr in attrs_list:
            if attr.startswith(search_key):
                attrs_out.append(search_key + value)
            else:
                attrs_out.append(attr)
        self.attributes = delimiter.join(attrs_out)

    def full_gfflist(self):
        list_out = [self.gffstring()]
        for child in self.children:
            list_out += child.full_gfflist()
        return list_out

    def full_gffstring(self):
        return '\n'.join(self.full_gfflist())

    def add_child(self, child):
        #todo, check type and ID
        if isinstance(child, GFFLine):
            self.children.append(child)

    def swap_strand(self):
        swap = {'+': '-', '-': '+'}
        self.strand = swap[self.strand]
        for child in self.children:
            child.strand = swap[child.strand]
        return swap

    def add_to_tree(self, tree):
        if not isinstance(tree, intervaltree.IntervalTree):
            raise ValueError("expected class IntervalTree, got {}".format(type(tree)))
        start = self.start
        end = self.end + 1  # to non-inclusive end point
        mdata = MinimalData(self.strand, self.region_type)
        data = mdata.as_dict()
        tree[start:end] = data


class MinimalData:
    cds = 'CDS'
    intron = 'intron'
    exon = 'exon'
    intergenic = 'intergenic'
    no_info = 'no_info'
    utr = 'utr'
    coding_start = 'coding_start'
    coding_stop = 'coding_stop'
    transcription_start = 'transcription_start'
    transcription_stop = 'transcription_stop'
    genic = 'genic'
    erroneous = 'erroneous'
    allowed_region_types = [cds, intron, exon, intergenic, genic, no_info, utr, coding_start, coding_stop,
                            transcription_start, transcription_stop, erroneous]

    plus_strand = '+'
    minus_strand = '-'
    unknown_strand = '.'
    allowed_strand_types = [plus_strand, minus_strand, unknown_strand]

    def __init__(self, strand, region_type):
        if region_type not in MinimalData.allowed_region_types:
            raise ValueError("region type: {}, not in allowed values {}".format(region_type,
                                                                                MinimalData.allowed_region_types))
        if strand not in MinimalData.allowed_strand_types:
            raise ValueError("strand type: {}, not in allowed values {}".format(strand,
                                                                                MinimalData.allowed_strand_types))
        self.region_type = region_type
        self.strand = strand

    def as_dict(self):
        return {'strand': self.strand,
                'type': self.region_type}


def test_commented_gffline():
    line = '#gff something'
    gl = GFFLineMrna(line)
    assert gl.comment == line
    assert gl.strand is None


def test_ncbi_gffline():
    line = """NC_003070.9\tRefSeq\tCDS\t3760\t3913\t.\t+\t0\tID=cds0;Parent=rna0;Dbxref=Araport:AT1G01010,TAIR:AT1G01010,GeneID:839580,Genbank:NP_171609.1;Name=NP_171609.1;Note=NAC domain containing protein 1 (NAC001)%3B FUNCTIONS IN: sequence-specific DNA binding transcription factor activity%3B INVOLVED IN: multicellular organismal development%2C regulation of transcription%3B LOCATED IN: cellular_component unknown%3B EXPRESSED IN: 7 plant structures%3B EXPRESSED DURING: 4 anthesis%2C C globular stage%2C petal differentiation and expansion stage%3B CONTAINS InterPro DOMAIN/s: No apical meristem (NAM) protein (InterPro:IPR003441)%3B BEST Arabidopsis thaliana protein match is: NAC domain containing protein 69 (TAIR:AT4G01550.1)%3B Has 2503 Blast hits to 2496 proteins in 69 species: Archae - 0%3B Bacteria - 0%3B Metazoa - 0%3B Fungi - 0%3B Plants - 2502%3B Viruses - 0%3B Other Eukaryotes - 1 (source: NCBI BLink).;gbkey=CDS;gene=NAC001;inference=similar to RNA sequence%2C mRNA:INSD:BT001115.1%2CINSD:AF439834.1%2CINSD:AK226863.1;orig_transcript_id=gnl|JCVI|mRNA.AT1G01010.1;product=NAC domain containing protein 1;protein_id=NP_171609.1
"""
    gl = GFFLine(line)
    restored_line = gl.full_gffstring()

    assert restored_line == line.rstrip()
    assert gl.strand == '+'
    assert gl.start == 3760
    assert gl.end == 3913
    assert gl.get_attribute('Parent') == 'rna0'

    gl.change_attribute('Dbxref', 'none')
    assert gl.full_gffstring() != line.rstrip()


class CoordinateError(Exception):
    pass


class TreelessInterval:
    def __init__(self, start, stop, data):
        if start >= stop:
            raise CoordinateError('start: {} not less than end: {} with data {}'.format(start, stop, data))
        self.start = start
        self.stop = stop
        self.data = data

    def __repr__(self):
        return 'TreelessInterval({}, {}, {})'.format(self.start, self.stop, self.data)


class GFFLineMrna(GFFLine):
    def __init__(self, gffline=None, *initial_data, **kwargs):
        super(GFFLineMrna, self).__init__(gffline=gffline, initial_data=initial_data, kwargs=kwargs)
        #GFFLine.__init__(self=self, gffline=gffline, initial_data=initial_data, kwargs=kwargs)  # python 2 compatible

        self.transcription_start = None
        self.transcription_stop = None
        self.coding_start = None
        self.coding_stop = None
        self.exons = []
        self.introns = []
        self.exons_coding_bits = []
        self.exons_non_coding_bits = []
        self.region_type = MinimalData.genic

    def process_children(self):
        try:
            self._consistency_check()
        except InconsistencyError:
            self.region_type = MinimalData.erroneous
        self._set_exons(consistency_check=False)
        self._set_introns(consistency_check=False)
        try:
            self._set_coding(consistency_check=False)
            self._set_non_coding(consistency_check=False)  # will fail if prev doesn't run
        except NoCDSFoundError:
            self.region_type = MinimalData.erroneous
        self._set_transcription_borders(consistency_check=False)
        try:
            self._consistency_check()
        except InconsistencyError:
            self.region_type = MinimalData.erroneous

    def _consistency_check(self):
        """double check things from parsing gff that code assumes are true"""
        if not self.children:
            raise NotSetupError("can't check consistency when no children have been added")

        for x in self.children:
            if not isinstance(x, GFFLine):
                raise ValueError("expected GFFLine, observed {}".format(type(x)))

        if not all([x.strand == self.strand for x in self.children]):
            e_str = "strand of all children does not match parent at {}".format(self)
            logging.debug(e_str)
            raise InconsistencyError(e_str)

        for alist in [self.exons, self.introns, self.exons_non_coding_bits, self.exons_coding_bits]:
            if not all([x.data == self.strand for x in alist]):
                e_str = "strand of TreelessInterval does not match parent at {}".format(self)
                logging.debug(e_str)
                raise InconsistencyError(e_str)

        if self.strand not in ['-', '+']:
            raise ValueError("unknown strand {}".format(self.strand))
        # todo, add within range testing

    def _set_exons(self, exon='exon', consistency_check=True):
        if consistency_check:
            self._consistency_check()
        if not self.children:
            raise NotSetupError("can't calculate exons when no children have been added")

        self.exons = []
        for child in self.children:
            if child.type == exon:
                if not (isinstance(child.start, int) and isinstance(child.end, int)):
                    raise ValueError('int expected for .start and .end attributes of GFFLineMrna')

                if child.start > child.end:  # hopefully this never occurs, but I don't trust gff 'standards'
                    child.start, child.end = child.end, child.start  # force correct sort
                    print('warning, start/end swapped at {}'.format(child))

                interval = TreelessInterval(child.start, child.end + 1, child.strand)  # +1 for non-inclusive end point
                self.exons.append(interval)
        self.exons.sort(key=lambda x: x.start)

    def _set_introns(self, consistency_check=True):
        if consistency_check:
            self._consistency_check()
        if not self.exons:
            raise NotSetupError("can't calculate introns until exons have been added, use self.set_exons()")

        self.introns = []
        for i in range(len(self.exons) - 1):
            ex_before = self.exons[i]
            ex_after = self.exons[i + 1]
            try:
                interval = TreelessInterval(ex_before.stop, ex_after.start, ex_before.data)
                self.introns.append(interval)
            except CoordinateError as e:
                # stupid, but not directly breaking things
                if ex_before.stop == ex_after.start:
                    logging.debug("Touching exons at mrna {}: {}".format(self.get_attribute('ID'), str(e)))
                # anything else is an impossible definition of a gene, and should be investigated
                else:
                    self.region_type = MinimalData.erroneous
                    err_msg = "Imposible coordinates at mrna {}: {}".format(self.get_attribute('ID'), str(e))
                    logging.critical(err_msg)

    def _set_transcription_borders(self, consistency_check=True):
        if consistency_check:
            self._consistency_check()
        if not self.exons:
            raise NotSetupError("can't calculate transcription start and stop until exons have been added")

        if self.strand == "+":
            start = self.exons[0].start  # the first bp of exon
            end = self.exons[-1].stop  # the first bp of intergenic region
        elif self.strand == '-':
            start = self.exons[-1].stop - 1  # the first bp of exon (reading reverse)
            end = self.exons[0].start - 1  # the first bp of intergenic region (reading reverse)
        else:
            raise ValueError("unknown strand creeped in {}".format(self.strand))
        self.transcription_start = TreelessInterval(start, start + 1, self.strand)
        self.transcription_stop = TreelessInterval(end, end + 1, self.strand)

    def _set_coding(self, cds='CDS', exon='exon', consistency_check=True):
        if consistency_check:
            self._consistency_check()
        # allow cds to be child of mRNA or of exon
        if not self.children:
            raise NotSetupError("can't calculate cds when no children have been added")

        self.exons_coding_bits = []
        cds = [x for x in self.children if x.type == cds]  # gets cds of they are child to mRNA
        if not cds:
            # gets cds if they are child to exons
            exons = [x for x in self.children if x.type == exon]
            if not exons:
                raise NoCDSFoundError("no CDS found, and no exons in fallback attempt at {}, with child types {}".format(
                    self.get_attribute('ID'),
                    [x.types for x in self.children]
                ))
            for exn in exons:
                cds += [x for x in exn.children if x.type == cds]
            if not cds:
                # raise trouble-shoot able error with available types printed out, if still no cds identified
                children_types = [x.type for x in self.children]
                sub_exon_types = [x.type for x in exn.children]
                e_str = "no CDS identified under {}, with types {}, and sub-last-exon types {} on {}".format(
                    self.get_attribute('ID'),
                    children_types,
                    sub_exon_types,
                    self.seqid
                )
                logging.debug(e_str)
                raise NoCDSFoundError(e_str)

        for coding_bit in cds:
            interval = TreelessInterval(coding_bit.start, coding_bit.end + 1, coding_bit.strand)
            self.exons_coding_bits.append(interval)
            self.exons_coding_bits.sort(key=lambda x: x.start)

        if self.strand == '+':
            start = self.exons_coding_bits[0].start  # first coding
            stop = self.exons_coding_bits[-1].stop  # first not coding (UTR or intergenic as it may be)
        elif self.strand == '-':
            start = self.exons_coding_bits[-1].stop - 1  # first coding (rev)
            stop = self.exons_coding_bits[0].start - 1  # first not coding (rev)
        else:
            raise ValueError("invalid strand {}".format(self.strand))

        self.coding_start = TreelessInterval(start, start + 1, self.strand)
        self.coding_stop = TreelessInterval(stop, stop + 1, self.strand)

    def _set_non_coding(self, consistency_check=True):
        if consistency_check:
            self._consistency_check()
        if not self.exons_coding_bits:
            raise NotSetupError("cds must be defined to calculate UTR")
        if not self.exons:
            raise NotSetupError("exons must be defined to calculate UTR")

        self.exons_non_coding_bits = []

        tree = intervaltree.IntervalTree()  # todo, since I've imported intervaltree now, use for all?
        # add all exons to tree
        for exn in self.exons:
            tree[exn.start:exn.stop] = exn.data
        # cut out all exon bits that are coding
        for cds in self.exons_coding_bits:
            tree.chop(cds.start, cds.stop)
        # that which remains is UTR (AKA non coding bit)
        for real_interval in tree:
            interval = TreelessInterval(real_interval.begin, real_interval.end, real_interval.data)
            self.exons_non_coding_bits.append(interval)
        self.exons_non_coding_bits.sort(key=lambda x: x.start)

    def swap_strand(self):
        #swap = super(GFFLineMrna, self).swap_strand()
        swap = GFFLine.swap_strand(self=self)  # py2
        for alist in [self.exons, self.introns, self.exons_non_coding_bits, self.exons_coding_bits]:
            for item in alist:
                item.data = swap[item.data]
        if self.coding_start is not None:
            self._set_coding()
        if self.transcription_stop is not None:
            self._set_transcription_borders()

    @staticmethod
    def sub_2_tree(treeless, region_type, tree):
        if treeless is not None:  # should only be set to none when it really wasn't possible to determine value
            md = MinimalData(treeless.data, region_type)
            tree[treeless.start:treeless.stop] = md.as_dict()

    def sub_all_2_tree(self, alist, region_type, tree):
        for treeless_interval in alist:
            self.sub_2_tree(treeless_interval, region_type, tree)

    def add_to_tree(self, tree):
        super(GFFLineMrna, self).add_to_tree(tree)
        #GFFLine.add_to_tree(self, tree)  # py2
        self.sub_all_2_tree(self.exons, MinimalData.exon, tree)
        self.sub_all_2_tree(self.introns, MinimalData.intron, tree)
        self.sub_all_2_tree(self.exons_coding_bits, MinimalData.cds, tree)
        self.sub_all_2_tree(self.exons_non_coding_bits, MinimalData.utr, tree)

        self.sub_2_tree(self.coding_start, MinimalData.coding_start, tree)
        self.sub_2_tree(self.coding_stop, MinimalData.coding_stop, tree)
        self.sub_2_tree(self.transcription_start, MinimalData.transcription_start, tree)
        self.sub_2_tree(self.transcription_stop, MinimalData.transcription_stop, tree)

def test_mrna_parsing_introns():
    line = "NC_003070.9\tRefSeq\tmRNA\t0\t549\t.\t+\t.\tID=rna0;Parent=gene0;transcript_id=NM_099983.2"
    child0 = "NC_003070.9\tRefSeq\texon\t0\t99\t.\t+\t.\tID=id0;Parent=rna0;transcript_id=NM_099983.2"
    child1 = "NC_003070.9\tRefSeq\texon\t200\t299\t.\t+\t.\tID=id1;Parent=rna0;transcript_id=NM_099983.2"
    child2 = "NC_003070.9\tRefSeq\texon\t500\t549\t.\t+\t.\tID=id2;Parent=rna0;transcript_id=NM_099983.2"
    childfail = "NC_003070.9\tRefSeq\texon\t505\t549\t.\t-\t.\tID=id3;Parent=rna0;transcript_id=NM_099983.2"
    gl = GFFLineMrna(line)
    cl0, cl1, cl2, clfail = [GFFLine(x) for x in [child0, child1, child2, childfail]]
    gl.children += [cl0, cl1, cl2]

    # test that _set_introns() method produces expected introns in plus strand
    gl._set_exons()
    gl._set_introns()
    expected = [TreelessInterval(100, 200, '+'), TreelessInterval(300, 500, '+')]
    assert len(expected) == len(gl.introns)
    for i in range(len(expected)):
        # test that all attributes of obj have same value
        assert expected[i].__dict__ == gl.introns[i].__dict__

    # test that it produces expected introns in minus strand
    for x in expected:
        x.data = '-'

    gl.swap_strand()

    assert gl.introns[0].data == '-'
    assert len(expected) == len(gl.introns)
    for i in range(len(expected)):
        # test that all attributes of obj have same value
        assert expected[i].__dict__ == gl.introns[i].__dict__

    # test that it catches invalid introns
    gl.add_child(clfail)  # causes invalid (overlapping) exon combo for one transcript
    gl._set_exons()

    #with pytest.raises(CoordinateError):
    gl._set_introns()
    assert gl.region_type == MinimalData.erroneous

def test_mrna_transcription_borders():
    line = "NC_003070.9\tRefSeq\tmRNA\t1\t549\t.\t+\t.\tID=rna0;Parent=gene0;transcript_id=NM_099983.2"
    child0 = "NC_003070.9\tRefSeq\texon\t1\t99\t.\t+\t.\tID=id0;Parent=rna0;transcript_id=NM_099983.2"
    child1 = "NC_003070.9\tRefSeq\texon\t200\t549\t.\t+\t.\tID=id1;Parent=rna0;transcript_id=NM_099983.2"
    gl = GFFLineMrna(line)
    gl.children = [GFFLine(x) for x in [child0, child1]]
    gl._set_exons()
    gl._set_transcription_borders()
    assert gl.transcription_start.start == 1
    assert gl.transcription_stop.start == 550

    gl.swap_strand()
    assert gl.transcription_stop.start == 0
    assert gl.transcription_start.start == 549


# todo, test the utr findy bit!
def test_mrna_cds_borders():
    mrnaline = "NC_003070.9\tRefSeq\tmRNA\t1\t549\t.\t+\t.\tID=rna0;Parent=gene0;transcript_id=NM_099983.2"

    childlines = ["NC_003070.9\tRefSeq\texon\t1\t99\t.\t+\t.\tID=id0;Parent=rna0;transcript_id=NM_099983.2",
                  "NC_003070.9\tRefSeq\texon\t200\t300\t.\t+\t.\tID=id1;Parent=rna0;transcript_id=NM_099983.2",
                  "NC_003070.9\tRefSeq\texon\t500\t549\t.\t+\t.\tID=id2;Parent=rna0;transcript_id=NM_099983.2",
                  "NC_003070.9\tRefSeq\tCDS\t15\t99\t.\t+\t.\tID=cds0;Parent=rna0;transcript_id=NM_099983.2",
                  "NC_003070.9\tRefSeq\tCDS\t200\t300\t.\t+\t.\tID=cds0;Parent=rna0;transcript_id=NM_099983.2",
                  "NC_003070.9\tRefSeq\tCDS\t500\t509\t.\t+\t.\tID=cds0;Parent=rna0;transcript_id=NM_099983.2"]
    mrna = GFFLineMrna(mrnaline)
    mrna.children = [GFFLine(x) for x in childlines]
    mrna.process_children()
    assert mrna.coding_start.start == 15
    assert mrna.coding_stop.start == 510
    assert len(mrna.exons_non_coding_bits) == 2
    assert mrna.exons_non_coding_bits[0].stop == 15
    assert mrna.exons_non_coding_bits[1].start == 510

    mrna.swap_strand()
    assert mrna.coding_start.start == 509
    assert mrna.coding_stop.start == 14
    assert len(mrna.exons_non_coding_bits) == 2
    assert mrna.exons_non_coding_bits[0].stop == 15
    assert mrna.exons_non_coding_bits[1].start == 510


class InconsistencyError(Exception):
    pass


class NoCDSFoundError(Exception):
    pass


class NotSetupError(Exception):
    pass


class GFFLineGene(GFFLine):

    def __init__(self, gffline=None, *initial_data, **kwargs):
        #super(GFFLineGene, self).__init__(gffline, initial_data, kwargs)
        GFFLine.__init__(self, gffline, initial_data, kwargs)  # py2
        self.region_type = MinimalData.genic

    def process_children(self):
        if not self.children:
            raise NotSetupError("cannot process children of GFFLineGene before they are added")
        for child in self.children:
            if not isinstance(child, GFFLineMrna):
                raise ValueError("expected class GFFLineMrna for child of GFFLineGene, got {}\n{}".format(type(child),
                                                                                                          child))

            child.process_children()

    def add_to_tree(self, tree):
        super(GFFLineGene, self).add_to_tree(tree)
        for child in self.children:
            child.add_to_tree(tree)


class GFFReader:
    def __init__(self):  # todo, pass parameters? attribute format
        self.comment = '#'
        self.gene = 'gene'
        self.mRNA = 'mRNA'  # 'transcript' also occur (in NCBI for no-cds alternatives of gen. protein coding genes)
        self.transcript = 'transcript'
        self.known_losses_at = ['pseudogene']
        self.attr_delim=';'
        #self.attr_eq='=' # todo
        self.attr_id = 'ID'
        self.attr_parent = 'Parent'
        # todo, any further formatting expectations

    @property
    def entry_subclasses(self):
        return {
            self.gene: GFFLineGene,
            self.mRNA: GFFLineMrna
        }

    def read_gfffile(self, gfffile):
        with open(gfffile) as f:
            i = 0
            for line in f:
                if not line.startswith(self.comment):
                    gffline = GFFLine(line)  # todo, pass **kwargs to here or anything?
                    # overwrite with subclass of GFFLine if it has a type specific subclass available
                    if gffline.type in self.entry_subclasses:
                        gffline = self.entry_subclasses[gffline.type](line)  # todo, this seems ugly, better way?

                    yield gffline
                i += 1
                if not i % 200000:
                    logging.debug('reading gfffile {}, passed line {}'.format(gfffile, i))
        logging.debug('read gff file {}'.format(gfffile))

    def recursive_cluster(self, gfflines, dump_at=None, check_at=None, check_still_ok=None):
        if dump_at is None:
            dump_at = self.gene

        if check_at is None:
            check_at = self.mRNA
        if check_still_ok is None:
            check_still_ok = [check_at, self.transcript]

        check_count0 = 0
        sorter = {}
        logging.debug('right before first pass, across gfflines, type {}'.format(gfflines))
        for entry in gfflines:
            # "id" is a bit of a joke, since for categories w/o children, they aren't unique x_x
            entry_id = entry.get_attribute(self.attr_id)
            try:
                sorter[entry_id].append(entry)
            except KeyError:
                #logging.debug("handling key error")
                sorter[entry_id] = [entry]
            if entry.type == check_at:
                check_count0 += 1
        logging.debug('half way through recursive cluster')
        # unfortunately one can't count on them being sorted/parents being in first, so take a second loop
        trans_splice_count = 0
        for key in sorter:
            entries = sorter[key]
            for entry in entries:
                #entry = sorter[key]
                parent_id = entry.get_attribute(self.attr_parent)
                if parent_id is not None:
                    parent_list = sorter[parent_id]
                    if len(parent_list) > 1:
                        try:
                            self.resolve_split_parent(parent_list)
                            assert len(parent_list) == 1  # make sure it worked
                            parent = parent_list[0]
                        except TransSplicingError:
                            # don't modify original list or it changes in sorter
                            tmp_parent_list = [x for x in parent_list if x.strand == entry.strand]
                            parent = tmp_parent_list[0]
                            logging.debug('Warning: multiple trans-spliced parts boiled '
                                          'down to {}, first chosen {}'.format(len(tmp_parent_list), parent_list))
                            trans_splice_count += 1
                        except ValueError as e:
                            logging.critical('marking {} and all parents as erroneous because: {}'.format(
                                entry.get_attribute('ID'), e
                            ))
                            for parent in parent_list:
                                parent.region_type = MinimalData.erroneous
                            parent = parent_list[0]
                    else:
                        parent = parent_list[0]
                    parent.add_child(entry)

        if trans_splice_count:
            logging.warning("{} non-mergable trans-splice events seen, "
                            "always took first with matching strand".format(trans_splice_count))

        # and a final run to get back out things organized by dump_at (genes)
        out = []
        check_count1 = 0
        known_lost = 0
        child_keys = {}
        for key in sorter:
            for raw_entry in sorter[key]:
                try:
                    entry = self.maybe_keep(raw_entry, dump_at, check_at)
                except UnmatchedChildrenError:
                    #print("Warning, swallowing non-fatal error: " + str(e))
                    entry = copy.deepcopy(raw_entry)
                    children = []
                    for child in entry.children:
                        try:
                            child_keys[child.type] += 1
                        except KeyError:
                            child_keys[child.type] = 1
                        if child.type == check_at:
                            children.append(child)  # keep only the expected
                        # mark as erroneous if not in [mRNA, transcript] (or similar / as set)
                        elif child.type not in check_still_ok:
                            entry.region_type = MinimalData.erroneous
                            logging.debug('marked {} as erroneous because of unrecognized child.type {}'.format(
                                entry.get_attribute('ID'), child.type
                            ))
                    entry.children = children
                if entry is not None:
                    out.append(entry)
                    for child in out[-1].children:
                        if child.type == check_at:
                            check_count1 += 1
                else:
                    if raw_entry.type in self.known_losses_at:
                        for child in raw_entry.children:
                            if child.type == check_at:
                                known_lost += 1

        if len(child_keys.keys()) > 1:
            s = "non-homgenous {} found, with summed composition of {}".format(check_at, child_keys)
            for x in child_keys.keys():
                if x not in check_still_ok:
                    logging.warning("can't handle {} mixed with {}, marked as erroneous.\n{}".format(
                        x, check_at, s
                    ))
            logging.info(s)

        if check_count0 != (check_count1 + known_lost):
            err = "parsing trouble: input had {} {}".format(check_count0, check_at)
            err2 = "output had {} accounted for ({} {} + {} known-losses from {})".format(check_count1 + known_lost,
                                                                                          check_count1,
                                                                                          check_at,
                                                                                          known_lost,
                                                                                          self.known_losses_at)
            raise ValueError("{}\n{}".format(err, err2))

        return out

    def resolve_split_parent(self, parent_list):
        """awful little helper to deal with genes that have been split into parts for no reasonable reason"""
        still_ok = True
        parts = [x.get_attribute('part') for x in parent_list]
        expected_ns = [int(x[-1]) for x in parts]  # part always formated s.t. like "part=1/2", "part=2/2"
        # make sure part number is self consistent
        if not all([x == expected_ns[0] for x in expected_ns[1:]]):
            still_ok = False
        expected_n = expected_ns[0]
        if len(parent_list) != expected_n:
            still_ok = False
        # make sure chromosome/molecule matches
        if not all([x.seqid == parent_list[0].seqid for x in parent_list[1:]]):
            still_ok = False
        # make sure strand matches
        if not all([x.strand == parent_list[0].strand for x in parent_list[1:]]):
            if 'trans-splic' not in parent_list[0].attributes:
                raise ValueError("multiple parents, multiple strands, not marked as trans-splice x_x\n{}".format(
                    parent_list
                ))
            else:
                raise TransSplicingError

        if still_ok:
            start = min([x.start for x in parent_list])
            end = max([x.end for x in parent_list])
            parent_list[0].start = start
            parent_list[0].end = end
            while len(parent_list) > 1:
                parent_list.pop()
            logging.debug('multiple parts boiled down to just one {}'.format(parent_list))
            return parent_list
        else:
            raise ValueError("multiple parents found and not self consistent as 'part's of one parent :-( {}".format(
                parent_list
            ))

    def maybe_keep(self, entry, dump_at, check_at):
        if entry.type == dump_at:
            children_check_out = [child.type == check_at for child in entry.children]
            if all(children_check_out):
                return copy.deepcopy(entry)
            elif any(children_check_out):
                raise UnmatchedChildrenError("parsing trouble, non-homogenous {} at {}".format(check_at, entry))
            else:
                return None
        else:
            return None


class UnmatchedChildrenError(Exception):
    pass


class TransSplicingError(Exception):
    pass


def test_split_paret():
    # working correctly
    lines = ["NC_000932.1\tRefSeq\tgene\t69611\t69724\t.\t-\t.\tID=gene38119;Dbxref=GeneID:1466250;Name=rps12;exception=trans-splicing;gbkey=Gene;gene=rps12;gene_biotype=protein_coding;locus_tag=ArthCp001;part=1/2",
             "NC_000932.1\tRefSeq\tgene\t97999\t98793\t.\t-\t.\tID=gene38119;Dbxref=GeneID:1466250;Name=rps12;exception=trans-splicing;gbkey=Gene;gene=rps12;gene_biotype=protein_coding;locus_tag=ArthCp001;part=2/2"]
    gfflines = [GFFLine(x) for x in lines]
    gfflines1 = copy.deepcopy(gfflines)
    gffreader = GFFReader()
    resolved = gffreader.resolve_split_parent(gfflines1)
    assert len(resolved) == 1
    assert resolved[0].start == 69611
    assert resolved[0].end == 98793
    # failing correctly
    # list does not match length expected from part
    gfflines2 = copy.deepcopy(gfflines)
    for i in range(2):
        gfflines2[i].change_attribute('part', '{}/3'.format(i + 1))

    with pytest.raises(ValueError):
        gffreader.resolve_split_parent(gfflines2)
    # all parts not assigned to the same chromosome/seqid
    gfflines3 = copy.deepcopy(gfflines)
    gfflines3[0].seqid = 'anything_but_matching'
    with pytest.raises(ValueError):
        gffreader.resolve_split_parent(gfflines3)


def write_gfffile(fileout, gffentries):
    gffout = open(fileout, 'w')
    n = len(gffentries)
    if n > 0:
        for entry in gffentries[0:(n-1)]:
            try:
                gffout.write(entry.full_gffstring() + '\n')
            except Exception as e:
                print('------\n')
                print(e)
                print(entry.gffstring())
                print(entry.full_gffstring())

        gffout.write(gffentries[n-1].full_gffstring())


def write_gffentries(file_handle, gffentries):
    for entry in gffentries:
        file_handle.write(entry.full_gffstring() + '\n')
