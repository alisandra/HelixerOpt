__author__ = 'Alisandra Denton'

import genic_problems
import pytest
import intervaltree
import gffreader
import tensorflow as tf
import numpy as np
from tensor2tensor.data_generators import generator_utils
import random
import os
import pickle
import copy


def test_stepper_xy():
    # 1:1 y:x but only center half of y
    xy_stepper = genic_problems.StepperXY(x_step=1, x_per_y=1, x_width=64, y_width=32)
    assert xy_stepper.relative_y[0] == 16
    assert xy_stepper.max_x_pad == 16

    # 1:2 full coverage
    xy_stepper = genic_problems.StepperXY(x_step=1, x_per_y=2, x_width=64, y_width=32)
    assert xy_stepper.relative_y[0] == 0
    assert xy_stepper.max_x_pad == 0

    # 1:2 center half of y
    xy_stepper = genic_problems.StepperXY(x_step=1, x_per_y=2, x_width=64, y_width=16)
    assert xy_stepper.relative_y[0] == 8
    assert xy_stepper.max_x_pad == 16

    # catches invalid
    with pytest.raises(ValueError):
        genic_problems.StepperXY(x_step=1, x_per_y=1, x_width=64, y_width=164)

    with pytest.raises(ValueError):
        genic_problems.StepperXY(x_step=0, x_per_y=1, x_width=64, y_width=64)
    # todo, test odd numbers, or that it isn't fed odd numbers?


def test_stepper_xy_stepping():
    some_values = ((64, 1, 64, 32, 256, 4),
                   (200, 1, 64, 64, 256, 1),
                   (1, 1, 32, 16, 32, 17),
                   (12, 2, 64, 32, 256, 17),
                   (33, 4, 64, 16, 263, 7),
                   (512, 1, 2048, 1024, 2048, 3))

    for x_step, x_per_y, x_width, y_width, end, expected_length in some_values:
        xy_stepper = genic_problems.StepperXY(x_step=x_step, x_per_y=x_per_y, x_width=x_width, y_width=y_width)
        ranges = list(xy_stepper.make_ranges(end))
        # first range pair, y_vals, left coor
        assert ranges[0][1][0] == 0
        # last range pair, y_vals, right coor
        assert ranges[-1][1][1] <= end
        assert len(ranges) == expected_length


def test_gen_matched_intervals():
    # create and populate an overlapping interval tree
    t = intervaltree.IntervalTree()
    for i in range(1, 100, 10):
        t[i:(i+20)] = 'data{}'.format(i)

    t.split_overlaps()  # split so there will be matches

    int_gen = genic_problems.gen_matched_intervals(t, 0, 90)
    an_int = next(int_gen)
    assert len(an_int) == 1
    assert an_int[0].end == 11

    for an_int in int_gen:
        assert len(an_int) == 2


def test_intervals_with_precedence():
    t = intervaltree.IntervalTree()
    md1 = gffreader.MinimalData('+', gffreader.MinimalData.intron)
    md2 = gffreader.MinimalData('+', gffreader.MinimalData.intergenic)
    data1 = md1.as_dict()
    data2 = md2.as_dict()
    t[1:10] = data1
    t[1:10] = data2
    intervals = list(t[0:11])
    y_levels = [gffreader.MinimalData.intergenic, gffreader.MinimalData.intron,
                gffreader.MinimalData.utr, gffreader.MinimalData.cds]
    winner = genic_problems.interval_with_precedence(intervals, y_levels)
    assert winner is not None
    assert winner.data['type'] == gffreader.MinimalData.intron

    # add another
    md3 = gffreader.MinimalData('-', gffreader.MinimalData.utr)
    t[1:10] = md3.as_dict()

    intervals = list(t[0:11])
    winner = genic_problems.interval_with_precedence(intervals, y_levels)
    assert winner.data['type'] == gffreader.MinimalData.utr

    # and a different bit of tree
    md4 = gffreader.MinimalData('-', gffreader.MinimalData.cds)
    t[20:30] = md4.as_dict()
    t[20:30] = md2.as_dict()

    intervals = list(t[20:30])
    winner = genic_problems.interval_with_precedence(intervals, y_levels)
    assert winner.data['type'] == gffreader.MinimalData.cds


def test_data_in_matches_data_out():
    """Tests whether dna/gene annotation data is the same before and after sharding"""
    file_shards = ['test_data_in_matches_data_out_shard']

    for file_shard in file_shards:
        if os.path.exists(file_shard):
            os.remove(file_shard)

    sequence_in = ''
    for _ in range(2048):
        sequence_in += random.choice('atcg')

    print(len(sequence_in))
    tree = intervaltree.IntervalTree()

    cds = gffreader.MinimalData('+', gffreader.MinimalData.cds)
    utr = gffreader.MinimalData('+', gffreader.MinimalData.utr)
    intron = gffreader.MinimalData('+', gffreader.MinimalData.intron)
    intergenic = gffreader.MinimalData('+', gffreader.MinimalData.intergenic)

    # entire bg defaults to intergenic final len (512 + 128 = 640)
    tree[0:2048] = intergenic.as_dict()
    tree[512:1024] = utr.as_dict()  # len 512
    tree[1024:1280] = cds.as_dict()  # len 256
    tree[1280:1536] = intron.as_dict()  # len 256
    tree[1536:1792] = cds.as_dict()  # len 256
    tree[1792:1920] = utr.as_dict()  # len 128

    expected_colsums = iter(([512, 0, 512, 0],
                             [0, 256, 512, 256],
                             [128, 256, 128, 512]))

    expected_Ns = iter([512, 0, 512])

    # step function, expect (left, centered, right) for y
    xy_stepper = genic_problems.StepperXY(x_step=512, x_per_y=1, x_width=2048, y_width=1024)
    mh = genic_problems.MolHolder(sequence_in, tree, random_seed=1, xy_stepper=xy_stepper)

    # call subfunction once
    coor_ranges = list(xy_stepper.make_ranges(len(sequence_in)))
    print(coor_ranges)
    # write examples to file

    def example_gen():
        for x, y in mh.generate(coor_ranges):
            yield genic_problems.make_example_dict(x, y)

    e_gen = example_gen()

    generator_utils.generate_files(e_gen,
                                   file_shards)
    # prep re-import
    problem = genic_problems.GeneCallingProblemTest()
    simple_in_fn = bare_input_fn(problem, file_shards)
    tensors = simple_in_fn()

    # in memory copy of what to expect
    mh2 = copy.deepcopy(mh)
    set_gen = mh2.generate(coor_ranges)
    with tf.Session() as sess:
        for i in range(len(coor_ranges)):
            print('trying {}'.format(i))
            x_original, y_original = next(set_gen)  # numpy array of what to expect
            exp_col_sums = next(expected_colsums)
            exp_n = next(expected_Ns)
            # check what we can that the _input_ actually matches expectations
            assert np.allclose(np.sum(y_original, axis=0), np.array(exp_col_sums))
            assert np.isclose(np.sum(x_original[x_original == 0.25]), exp_n)

            x_reimported, y_reimported = sess.run(tensors)

            # check that reimported matches input
            try:
                assert np.allclose(x_original, x_reimported)
                assert np.allclose(y_original, y_reimported)
            except Exception as e:
                out = {'x_in': x_original, 'x_out': x_reimported, 'y_in': y_original, 'y_out': y_reimported}
                fileout = 'test.pkl'
                with(open(fileout, 'wb')) as f:
                    pickle.dump(out, f)
                print('wrote pickle to troubleshooting {}'.format(fileout))
                raise e

        print(coor_ranges)
        for file_shard in file_shards:
            if os.path.exists(file_shard):
                os.remove(file_shard)


def bare_input_fn(problem, file_shards):
    def in_fn():
        # input reader  # todo, if input function had it's own file, I could import it
        dataset = tf.contrib.data.TFRecordDataset(file_shards)

        # Map the parser over dataset, and batch results by up to batch_size
        dataset = dataset.map(lambda eg: problem.parser(eg, None, meta_info={}))
        dataset = dataset.batch(1)
        dataset = dataset.repeat()
        iterator = dataset.make_one_shot_iterator()
        tensors = iterator.get_next()
        return tensors
    return in_fn


def test_in_out_identity():
    """tests whether mock data is the same before and after sharding"""
    file_shards = ['test_in_out_id0']

    for file_shard in file_shards:
        if os.path.exists(file_shard):
            os.remove(file_shard)

    problem = genic_problems.GeneCallingProblemTest()

    total = 15

    def example_setup():
        for ii in range(total):
            x = np.zeros((problem.len_x, 4))
            y = np.zeros((problem.len_y, problem.number_y))
            x[ii, 0] = 1
            y[ii, 0] = 1
            yield x, y

    def example_gen():
        ex = example_setup()
        for x, y, in ex:
            out = genic_problems.make_example_dict(x, y)
            yield out

    examples_in = example_setup()
    ex_gen = example_gen()

    generator_utils.generate_files(ex_gen, file_shards)

    dataset = tf.data.TFRecordDataset(file_shards)

    # Map the parser over dataset, and batch results by up to batch_size
    dataset = dataset.map(lambda eg: problem.parser(eg, None))
    iterator = dataset.make_one_shot_iterator()
    tensors = iterator.get_next()
    with tf.Session() as sess:
        for i in range(total):
            x_original, y_original = next(examples_in)  # numpy array of what to expect

            x_reimported, y_reimported = sess.run(tensors)

            # check that reimported matches input
            try:
                assert np.allclose(x_original, x_reimported)
                assert np.allclose(y_original, y_reimported)
            except Exception as e:
                out = {'x_in': x_original, 'x_out': x_reimported,
                       'y_in': y_original, 'y_out': y_reimported}
                fileout = 'test2.pkl'
                with(open(fileout, 'wb')) as f:
                    pickle.dump(out, f)
                print('wrote pickle to troubleshooting {}'.format(fileout))

                print('argmax x: {}, y: {} at i={}'.format(np.argmax(x_reimported[:, 0]),
                                                           np.argmax(y_reimported[:, 0]),
                                                           i))
                print('shapes x: {}, y: {}'.format(x_reimported.shape, y_reimported.shape))
                raise e
    for file_shard in file_shards:
        if os.path.exists(file_shard):
            os.remove(file_shard)

