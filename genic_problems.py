from __future__ import division

__author__ = 'Alisandra Denton'

from tensor2tensor.data_generators import generator_utils
from tensor2tensor.utils import metrics

import tensorflow as tf
import numpy as np

import random
import os
import intervaltree
import copy
import logging
from multiprocessing import Process #, cpu_count
import time

from dna_helpers import seq_to_row, row_to_array, fasta2seqs, padded_subseq, reverse_complement
from mod_problem import ModProblem, serialize_from_numpy
import gffreader


### gene expression problems ###
class GeneCallingProblem(ModProblem):
    """Base Problem for gene calling/structural annotation.

    Transforms input .gff & .fa from one directory (with matching names) into 1-hot vectors of [C,A,T,G] for features
    and [intergenic, intron, utr, cds] for targets."""

    TRAIN = 'train'
    TEST = 'test'
    XVAL = 'val'

    def __init__(self, was_reversed=False, was_copy=False):
        super().__init__(was_reversed, was_copy)  # likely python2 trouble
        self._threads = 1
    # todo, should some of these be implemented first in subclasses? NotImplementedError
    @property
    def len_x(self):
        return 2048

    @property
    def number_x(self):
        return 4  # C,A,T,G (in that order)

    @property
    def len_y(self):
        return 1024  # predictions 1 / bp, but centered in total DNA provided

    @property
    def y_levels(self):
        return [gffreader.MinimalData.intergenic, gffreader.MinimalData.intron,
                 gffreader.MinimalData.utr, gffreader.MinimalData.cds]

    @property
    def number_y(self):
        out = 4
        assert out == len(self.y_levels)
        return out  # intergenic, intron, utr, cds (in that order)

    @property
    def x_per_y(self):
        return 1  # don't change without making 'annotated_range_from_tree' take, and dynamically use as parameter

    @property
    def relative_x_step(self):
        return 50

    @property
    def label_shape(self):
        return [self.len_y, self.number_y]

    @property
    def pfx_seed(self):
        # final seed will be `specie_name + sequence_id + self.pfx_seed`
        # this allows it to be deterministically 'random' for a given species, regardless of the rest, unless overridden
        return 'puma'

    @property
    def directory_in(self):
        raise NotImplementedError

    @property
    def num_shards_dev_test(self):
        # todo, at some point if I'm cleaning input data, this could go back to 10 (AKA inherit from parent)
        return self.num_shards // 3

    @property
    def fasta_ending(self):
        return '.fa'

    @property
    def gff_ending(self):
        return '.gff'

    @property
    def multiprocess_generate(self):
        return True

    @property
    def num_generate_tasks(self):
        return self._threads

    @num_generate_tasks.setter
    def num_generate_tasks(self, n):
        n_cpus = 8 #cpu_count()
        if n < 1:
            n = 1
        elif n > n_cpus:
            n = n_cpus
        self._threads = n

    @property
    def evaluation_only(self):
        return False

    def generate_data(self, data_dir, tmp_dir='/tmp/', task_id=-1):
        threads = self.num_generate_tasks  # todo, dynamic
        sp_names = self.get_sp_names()

        three_set_names = [self.TRAIN, self.XVAL, self.TEST]
        out_file_paths = self.setup_file_paths(data_dir)
        all_dev = self.evaluation_only

        paths_to_shuffle = out_file_paths['train'] + out_file_paths['dev'] + out_file_paths['test']  # todo, still ugly
        jobs_list = self.divvy_up_jobs(sp_names,
                                       out_file_paths['train'],
                                       out_file_paths['dev'],
                                       out_file_paths['test'])
        # sometimes it is nice to have a non-multi threaded option
        if threads == 1:
            for job in jobs_list:
                logging.info("working with {}, with {} shards for train, dev, test respectively".format(
                    job[0],
                    [len(x) for x in job[1:]])
                )
                targ_sp = job[0]
                self.generate_dataset(targ_sp,
                                      job[1:],
                                      three_set_names,
                                      all_dev)
        else:
            processes = []
            for job in jobs_list:
                logging.info("working with {}, with {} shards for train, dev, test respectively".format(
                    job[0],
                    [len(x) for x in job[1:]])
                )
                targ_sp = job[0]
                processes.append(Process(target=self.generate_dataset,
                                         args=(targ_sp,
                                               job[1:],
                                               three_set_names,
                                               all_dev)))

            # start a set of processes
            running = []
            for i in range(min(threads, len(processes))):
                p = processes.pop()
                p.start()
                running.append(p)

            while running:
                time.sleep(9)
                # rm finished
                for i in range(len(running) - 1, -1, -1):
                    if not running[i].is_alive():
                        running.pop(i)
                # add more
                if processes:
                    try:
                        for i in range(threads - len(running)):
                            p = processes.pop()
                            p.start()
                            running.append(p)
                    except IndexError:
                        continue

        if self.shuffle_me:
            generator_utils.shuffle_dataset(paths_to_shuffle)

    def generate_dataset(self,
                         targ_sp,
                         three_file_sets,
                         three_set_names,
                         all_dev=False):
        logging.info("self.generate_dataset starting for targ_sp {}".format(targ_sp))
        mol_holders = self.import_files_to_mol_holders(targ_sp, all_dev=all_dev)
        logging.info("mol_holders successfully imported for {}".format(targ_sp))
        for outfiles, set_name in zip(three_file_sets, three_set_names):
            generator_utils.generate_files(self.dataset_generator(set_name, mol_holders),
                                           outfiles)
            logging.info("self.generate_dataset finished for targ_sp {}".format(targ_sp))

    def generate_dataset02(self, targ_sp, three_file_sets, three_set_names, all_dev=False):
        logging.info("generate_dataset02")
        for outfiles, set_name in zip(three_file_sets, three_set_names):
            generator_utils.generate_files(self.gen_dummy(100000), outfiles)

    @staticmethod
    def gen_dummy(n):
        for i in range(n):
            try:
                if i % 1000:
                    raise ValueError
            except ValueError:
                pass
            yield {'inputs': [0.], 'targets': [0.]}

    @staticmethod
    def dataset_generator(a_set,
                          mol_holders):
        logging.info("self.dataset_generator starting now for {}".format(a_set))
        for mh in mol_holders:
            for x, y in mh.generate_set(a_set):
                ex_dict = make_example_dict(x, y)
                # todo, test alllllll the things
                yield ex_dict
        logging.info("self.dataset_generator should raise stop iteration now for {}".format(a_set))

    def divvy_up_jobs(self, *all_files):
        n_out = min([len(x) for x in all_files])
        pre_out = [self.chunk_list(n_out, x) for x in all_files]
        out = list(map(list, zip(*pre_out)))
        return out

    @staticmethod
    def chunk_list(n, a_list):
        a_list = copy.deepcopy(a_list)
        out = [[] for _ in range(n)]
        i = 0
        while a_list:
            out[i].append(a_list.pop())
            i += 1
            if i == n:
                i = 0
        return out

    def get_sp_names(self):
        all_file_ins = os.listdir(self.directory_in)
        all_file_ins.sort()
        fasta_files = [x for x in all_file_ins if x.endswith(self.fasta_ending)]
        gff_files = [x for x in all_file_ins if x.endswith(self.gff_ending)]
        sp_names = [x.replace(self.gff_ending, '') for x in gff_files]
        sp_names2 = [x.replace(self.fasta_ending, '') for x in fasta_files]

        for i in range(max(len(sp_names), len(sp_names2))):
            if sp_names[i] != sp_names2[i]:
                raise NonMatchingFiles("cannot match up gff and fasta files in {}, "
                                       "name from gff {}, name from fa {}".format(self.directory_in,
                                                                                  sp_names[i],
                                                                                  sp_names2[i]))
        return sp_names

    def import_files_to_mol_holders(self, sp_names, all_dev=False):

        x_per_y = self.x_per_y
        y_width = self.len_y
        x_width = self.len_x
        xy_stepper = StepperXY(x_step=int(y_width * x_per_y * self.relative_x_step),
                               x_per_y=x_per_y, x_width=x_width, y_width=y_width)

        mol_holders = []
        for sp in sp_names:
            logging.info('importing from {}'.format(sp))
            # setup chromosome / sequence names to skip (e.g. Mt, Pt; must match fasta names)
            exclude = []
            exclude_file = '{}/{}.exclude'.format(self.directory_in, sp)
            if os.path.exists(exclude_file):
                with open(exclude_file) as f:
                    for line in f:
                        exclude.append(line.rstrip())
                logging.info('excluding the following sequences {}'.format(exclude))

            fasta_file = '{}/{}{}'.format(self.directory_in, sp, self.fasta_ending)
            gff_file = '{}/{}{}'.format(self.directory_in, sp, self.gff_ending)

            logging.debug('reading in fasta file {}'.format(fasta_file))
            fasta = fasta2seqs(fasta_file)
            logging.debug('reading in gff file {}'.format(gff_file))
            trees = load_gff_to_intervaltrees(gff_file)
            logging.debug('finished gff file{}'.format(gff_file))
            # check if we got matching keys
            try:
                keys_nearly_match = self.are_keys_compatible(fasta, trees, sp)
            except NonMatchingSequences:
                keys_nearly_match = self.reconstruct_key_match(fasta, trees, sp)
            except DataProcessingError as e:
                logging.critical(e)
                continue

            for seqid in fasta:
                tree = None

                if seqid not in exclude:
                    try:
                        tree = trees[seqid]
                    except KeyError as e:
                        # todo, this should be auto-detectable from gffreader at least, and maybe also marked in fa header
                        if keys_nearly_match:
                            logging.warning('Skipping seqid: {}, from sp: {}, on the assumption it is '
                                            'organellar'.format(seqid, sp))
                        else:
                            logging.debug('swallowing KeyError {}'.format(e))

                if tree is not None:
                    mh = MolHolder(seq=fasta[seqid],
                                   tree=tree,
                                   random_seed=sp + seqid + self.pfx_seed,
                                   xy_stepper=xy_stepper,
                                   y_levels=self.y_levels,
                                   sp=sp,
                                   all_dev=all_dev)
                    mol_holders.append(mh)

        return mol_holders

    @staticmethod
    def warn_missing_keys(sp, r_keys, o_keys, ref_name='gff-annotated', other_name='fasta'):
        r_only = r_keys.difference(o_keys)
        if len(r_only) > 0:
            max_show = min(len(r_only), 5)
            s = "Warning: at {}, {} of {} {} sequences lack {} sequences, e.g. {}".format(sp, len(r_only), len(r_keys),
                                                                                          ref_name, other_name,
                                                                                          list(r_only)[0:max_show])
            logging.warning(s)

    def are_keys_compatible(self, fasta, trees, sp):
        logging.info('checking whether IDs match between gff and fasta')
        # todo, maybe handle entirely non-ambiguous contains sorts of problems?
        f_keys = set(fasta.keys())
        t_keys = set(trees.keys())
        ok = False
        if len(t_keys) == len(f_keys):
            if f_keys.issubset(t_keys):  # perfect matches always ok
                ok = True
        elif 0 < (len(f_keys) - len(t_keys)) <= 2:
            if t_keys.issubset(f_keys):  # probably just the organnelles, also ok / warning elsewhere
                ok = True

        if not ok:
            if len(f_keys) == 0:
                raise DataProcessingError("No fasta sequences imported")
            if len(t_keys) == 0:
                raise DataProcessingError("No gff-annotated sequences with genes / mRNA imported")
            # check if this is beyond salvage
            if len(f_keys.intersection(t_keys)) == 0:
                e_str = "no sequence names could be matched between gff (e.g. {}) and fasta (e.g. {}) in {}".format(
                    list(sorted(t_keys))[0],
                    list(sorted(f_keys))[0],
                    sp
                )
                raise NonMatchingSequences(e_str)
            # warn if there were at least some matches
            self.warn_missing_keys(sp, t_keys, f_keys, ref_name='gff-annotated', other_name='fasta')
            self.warn_missing_keys(sp, f_keys, t_keys, ref_name='fasta', other_name='gff-annotated')

        return ok

    def reconstruct_key_match(self, fasta, trees, sp):
        logging.info("attempting to match up, non-identical sequence IDs between gff and fasta")
        f_keys = set(fasta.keys())
        t_keys = set(trees.keys())
        old2new = {}
        # for each tree key, does it have exactly one match?
        for key in t_keys:
            matches = [x for x in f_keys if key in x]
            if len(matches) == 1:
                # setup dict[old_key] = new_key
                old2new[key] = matches[0]
            elif len(matches) == 0:
                # pretending no match is ok, (a warning will be logged by are_keys_compatible) but seriously NCBI?
                logging.debug('no matches found for {} in fasta keys, e.g. {}'.format(
                    key, list(f_keys)[:min(10, len(f_keys))]
                ))
            else:
                raise NonMatchingSequences('could not identify unique match for {}, but instead got {}'.format(key,
                                                                                                               matches))
        # check we matched trees to _unique_ fasta keys
        if len(old2new.values()) != len(set(old2new.values())):
            raise NonMatchingSequences('could not uniquely match up tree_keys: {} and fasta_keys {}'.format(t_keys,
                                                                                                            f_keys))
        # change the name of tree keys to match the hits from fasta keys
        for key in old2new:
            trees[old2new[key]] = trees.pop(key)

        return self.are_keys_compatible(fasta, trees, sp)  # check output and lengths once more

    def example_reading_spec(self):
        data_fields = {
            'inputs': tf.FixedLenFeature([self.number_x * self.len_x], tf.float32),
            'targets': tf.FixedLenFeature([self.number_y * self.len_y], tf.float32)
        }

        return data_fields, None  # idk what the None is a placeholder for in real t2t land

    def preprocess_example(self, example, mode, unused_hparams):

        # restore original shapes
        example["targets"] = tf.reshape(example["targets"],
                                        [self.len_y, self.number_y])

        example["inputs"] = tf.reshape(example["inputs"], (-1, 4))

        if mode == tf.estimator.ModeKeys.TRAIN:
            # todo, mask input regions in train mode
            pass
            # todo, the following permutations are broken, and never worked, respectively. Fix.
            #example = self.maybe_reverse_complement(example)
            #example = self.maybe_0_out(example)

        return example

    @staticmethod
    def maybe_reverse_complement(example, prob=0.5):
        if random.random() <= prob:
            # three dimensions, for reverse, complement, and anyways length of 1 (so doesn't matter)
            # todo, please test this!
            example['inputs'] = tf.reverse(example['inputs'], [0, 1, 2])  # todo, test for genic!!
            example['targets'] = tf.reverse(example['targets'], [1])
        return example

    def parser(self, serialized_example, mode, meta_info=None):
        if meta_info is None:
            meta_info = {}

        data_fields, _ = self.example_reading_spec()
        features = tf.parse_single_example(serialized_example,
                                           features=data_fields)
        example = self.preprocess_example(features, mode, meta_info)
        features = example['inputs']
        labels = example['targets']
        return features, labels

    def eval_metrics(self):
        return [metrics.Metrics.LOG_POISSON, metrics.Metrics.R2]


# custom exceptions
class NonMatchingFiles(Exception):
    pass


class NonMatchingSequences(Exception):
    pass


class DataProcessingError(Exception):
    pass


# extra helpers / misc...
def make_example_dict(x, y):
    ex_dict = {
        'inputs': serialize_from_numpy(x),
        'targets': serialize_from_numpy(y)
    }
    return ex_dict


class MolHolder:
    def __init__(self, seq, tree, random_seed, xy_stepper, y_levels=None, sp='', all_dev=False):
        if not isinstance(xy_stepper, StepperXY):
            raise TypeError
        # todo, add species and sequence ID, so errors can be debugged
        self.padding = xy_stepper.max_x_pad  # add padding so we can predict y all the way to ends of molecule
        self.seq = seq
        self.tree = tree
        self.add_intergenic_background()
        self.y_levels = y_levels
        self.sets = xy_stepper.make_sets(random_seed=random_seed, end=len(self.seq), all_dev=all_dev)
        self.sp = sp

    def generate_set2(self, a_set):
        for x, y in self.generate(self.sets[a_set]):
            yield x, y

    def generate_set(self, a_set):
        #logging.info('generating from MolHolder for set {}'.format(a_set))
        number_errors = 0
        atg_errors = 0
        for x_coords, y_coords in self.sets[a_set]:
            pre_x = padded_subseq(self.seq, x_coords[0], x_coords[1])
            x = row_to_array(seq_to_row(pre_x, spread_evenly=True))
            try:
                y = annotated_range_from_tree(self.tree, y_coords[0], y_coords[1], self.seq, self.y_levels)
                yield x, y
            except AssertionError:
                number_errors += 1
            except StartNotATGError as e:
                logging.debug(str(e))
                atg_errors += 1

        if number_errors:
            logging.warning('{} erroneous examples skipped, set {}, sp {}, seq starting with {}'.format(
                number_errors,
                a_set,
                self.sp,
                self.seq[0:min(50, len(self.seq))])
            )
        if atg_errors:
            logging.warning('{} examples containing non-ATG start skipped, set {}, sp {}, seq starting with {}'.format(
                atg_errors,
                a_set,
                self.sp,
                self.seq[0:min(50, len(self.seq))])
            )

    def add_intergenic_background(self):
        minimal_data = gffreader.MinimalData('+', gffreader.MinimalData.intergenic)
        self.tree[0:len(self.seq)] = minimal_data.as_dict()

    def generate(self, coordinate_pairs):
        # here for testing purposes only atm
        #  todo, should be recombined so this + generate_set2 has full functionality (and name) of generate_set
        for x_coords, y_coords in coordinate_pairs:
            pre_x = padded_subseq(self.seq, x_coords[0], x_coords[1])
            x = row_to_array(seq_to_row(pre_x, spread_evenly=True))
            y = annotated_range_from_tree(self.tree, y_coords[0], y_coords[1], self.seq, self.y_levels)
            yield x, y


def load_gff_to_intervaltrees(gfffile, max_errors=100):
    interval_trees = {}

    gr = gffreader.GFFReader()
    gfflines = gr.read_gfffile(gfffile)
    try:
        logging.info('starting clustering {}'.format(gfffile))
        genes = gr.recursive_cluster(gfflines)
        logging.debug('finished clustering {}'.format(gfffile))
    except AssertionError as e:
        logging.warning('error at {}'.format(gfffile))
        print('at {}'.format(gfffile))
        raise e

    setup_errors = 0
    cds_errors = 0
    for gene in genes:
        try:
            gene.process_children()
        except gffreader.NotSetupError as e:
            logging.debug('ignoring NotSetupError {}'.format(e))
            setup_errors += 1
            # we expect some errors, e.g. due to trans-splice imperfect handling, but more than a few
            # is a sign the gff is not
            # being interpreted correctly
            if setup_errors > max_errors:
                logging.warning('Too many setup errors, marking remainder as erroneous, e.g. {}'.format(e))
                gene.region_type = gffreader.MinimalData.erroneous

        try:
            gene.add_to_tree(interval_trees[gene.seqid])
        except KeyError:
            #logging.debug('creating new interval tree {}'.format(gene.seqid))
            interval_trees[gene.seqid] = intervaltree.IntervalTree()
            gene.add_to_tree(interval_trees[gene.seqid])

    if setup_errors > 0:
        logging.warning('encountered {} setup errors'.format(setup_errors))
    if cds_errors > 0:
        logging.warning('encountered {} no-CDS errors'.format(cds_errors))

    return interval_trees


class StartNotATGError(Exception):
    pass


def annotated_range_from_tree(tree, start, end, sequence, y_levels=None):
    """create labeled (based on y_levels) onehot for range on intervaltree (with data['type'] matching y_levels)"""
    if y_levels is None:
        y_levels = [gffreader.MinimalData.intergenic, gffreader.MinimalData.intron,
                    gffreader.MinimalData.utr, gffreader.MinimalData.cds]

    y_out = np.zeros((end - start, len(y_levels)))

    intervals = tree[start:end]

    # skip examples were the TSS is marked but is not ATG

    start_positions = [x for x in intervals if x.data['type'] == gffreader.MinimalData.transcription_start]
    for start_pos in start_positions:
        strand = start_pos.data['strand']
        if strand == '+':
            start = start_pos.begin
            codon = sequence[start:(start + 3)].upper()
        elif strand == '-':
            start = start_pos.begin
            codon = sequence[(start - 2):(start + 1)].upper()
            try:
                codon = reverse_complement(codon)
            except KeyError as e:
                raise StartNotATGError('expected ATG, could not reverse complement with KeyError: {}'.format(e))
        else:
            raise ValueError('how did a non +- strand make it to here?')
        if codon != 'ATG':
                raise StartNotATGError('expected ATG, found {}, at {} on strand {}'.format(codon, start, strand))

    for interval in intervals:
        assert interval.data['type'] != gffreader.MinimalData.erroneous
    intervals = [x for x in intervals if x.data['type'] in y_levels]  # data previously set via gffreader.MinimalData
    subtree = intervaltree.IntervalTree(intervals)

    i = 0
    for matched_intervals in gen_matched_intervals(subtree, start, end):
        # todo, check for and handle any gaps?
        winner = interval_with_precedence(matched_intervals, y_levels)
        length = winner.end - winner.begin
        y_out[i:(i + length), y_levels.index(winner.data['type'])] = 1
        i += length

    assert np.sum(y_out) == y_out.shape[0]  # make sure everything was assigned an index
    return y_out
    # todo, test


def gen_matched_intervals(tree, begin, end):
    if not isinstance(tree, intervaltree.IntervalTree):
        raise TypeError

    # slice off and drop any out-of-range tails
    tree.slice(begin)
    tree.slice(end)
    tree = intervaltree.IntervalTree(tree[begin:end])
    # slice again so intervals overlap each other either entirely or not at all
    tree.split_overlaps()

    iter_tree = iter(sorted(tree))

    # match up intervals by .begin position
    prev_interval = next(iter_tree)
    out = [prev_interval]

    for interval in iter_tree:

        if prev_interval.begin != interval.begin:
            yield out  # yield in matching groups
            out = []

        out.append(interval)
        prev_interval = interval

    yield out


def interval_with_precedence(intervals, y_levels):
    sorted_intervals = list(intervals)
    sorted_intervals.sort(key=lambda x: y_levels.index(x.data['type']))
    if sorted_intervals is not None:
        out = sorted_intervals[-1]
    else:
        out = None
    return out


class StepperXY:
    def __init__(self, x_step, x_per_y, x_width, y_width):
        if not all([isinstance(x, int) for x in [x_step, x_per_y, x_width, y_width]]):
            raise ValueError("all parameters for StepperXY init must be integers")

        if x_width < (y_width * x_per_y):
            raise ValueError("cannot predict y beyond supplied x, what should this mean?")

        if x_step < 1:
            raise ValueError('x_step must be a positive integer')

        self.centered = True
        self.y_width = y_width
        self.x_width = x_width
        self.x_per_y = x_per_y
        self.x_step = x_step

        # derived from the others
        self.relative_x, self.relative_y, self.max_x_pad = self._relative()

    def make_ranges(self, end):
        # i is always start coordinate
        # start from negative padding for x (so y starts at 0)
        start = -self.max_x_pad
        # end at last full fit in padded sequence (seq_length + padding - x_width)
        # note that end itself (inclusive) should be an acceptable stopping point (not inclusive), hence +1
        stop = end + self.max_x_pad - self.x_width + 1

        for i in range(start, stop, self.x_step):
            x_out = [x + i for x in self.relative_x]
            y_out = [x + i for x in self.relative_y]
            yield x_out, y_out

    def _relative(self):
        x = (0, self.x_width)
        # centering y relative to x
        y_start = (self.x_width // self.x_per_y - self.y_width) // 2
        y = (y_start, y_start + self.y_width)

        full_x_pad = int(y_start * self.x_per_y)  # note how much chr would need padding to allow capture of all y
        return x, y, full_x_pad

    def make_sets(self, random_seed, end, all_dev=False):
        if all_dev:
            train_at_or_below = -1
            test_at_or_below = -1
        else:
            train_at_or_below = 0.6
            test_at_or_below = 0.8
        # for latter filling
        sets = {GeneCallingProblem.TRAIN: [],
                GeneCallingProblem.TEST: [],
                GeneCallingProblem.XVAL: []}

        random.seed(random_seed)  # should always be the same for the same species
        for x, y in self.make_ranges(end):
            r = random.random()
            a_set = GeneCallingProblem.XVAL
            if r <= train_at_or_below:
                a_set = GeneCallingProblem.TRAIN
            elif r <= test_at_or_below:
                a_set = GeneCallingProblem.TEST

            sets[a_set].append((x, y))

        return sets
