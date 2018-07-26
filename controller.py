from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__author__ = 'Alisandra Denton'

import os
import sys
import argparse
try:
    import _pickle as pickle
except ImportError:
    import pickle
import logging
import numpy as np
import tensorflow as tf
import json

# all of the following is used, just through `eval(str)`
# I know, shouldn't import * ... todo, figure out how to use t2t's registry
from tissue_problems_subclasses import *  
from genic_problems_subclasses import *
from fieldspec_problems_subclasses import *
from custom_models import *

# for previewing
def get_latest_checkpoint(train_dir):
    metas = os.listdir(train_dir)
    metas = [x for x in metas if x.endswith('.meta')]
    metas = sorted(metas)
    latest_meta = metas[-1]
    latest_checkpoint = latest_meta.replace('.meta', '')
    return train_dir + '/' + latest_checkpoint


def input_fn_2_feed_dict(x, y, session):
    # works for input (x, y) as produced by an input_fn created below
    # I suggest PREDICT mode, to avoid shuffling should later comparison be desireable
    # e.g. x, y = predict_input_fn()
    features, labels = session.run([x, y]) #subprocess.run()
    #labels = session.run(y)
    feed_dict = {x.name: features, y.name: labels}
    return feed_dict


# for everything
def input_fn(batch_size, mode, data_dir, problem):
    if mode == tf.estimator.ModeKeys.TRAIN: #ModeKeys.x; x= TRAIN, EVAL, PREDICT
        all_filenames = problem.training_filepaths(data_dir, problem.num_shards, shuffled=True)
    elif mode in [tf.estimator.ModeKeys.EVAL, tf.estimator.ModeKeys.PREDICT]:
        all_filenames = problem.dev_filepaths(data_dir, problem.num_shards_dev_test, shuffled=True)
    else:
        raise ValueError('only train and dev input_fn implemented so far')

    training_stats = problem.make_meta(data_dir)

    def fn():
        dataset = tf.data.TFRecordDataset(all_filenames)

        # Map the parser over dataset, and batch results by up to batch_size
        dataset = dataset.map(lambda eg: problem.parser(eg, mode, meta_info=training_stats))
        dataset = dataset.batch(batch_size)
        dataset = dataset.repeat()
        if mode != tf.estimator.ModeKeys.PREDICT:
            dataset = dataset.shuffle(1000)
        iterator = dataset.make_one_shot_iterator()

        features, labels = iterator.get_next()

        return features, labels
    return fn



def main_datagen(data_dir, problem):
    logging.info('starting data generation')
    problem.generate_data(data_dir)
    print('finished generating data in: ' + data_dir)


def main_predict(data_dir, train_dir, model_fn, problem, n=160, fileout=None):
    logging.info('starting prediction')
    if fileout is None:
        fileout = train_dir + '/predictions.pkl'

    pred_input_fn = input_fn(n, tf.estimator.ModeKeys.PREDICT, data_dir, problem)
    gt_input_fn = input_fn(n, tf.estimator.ModeKeys.PREDICT, data_dir, problem)

    model_params = {'labels_shape': problem.label_shape}

    nn = tf.estimator.Estimator(model_fn=model_fn, params=model_params, model_dir=train_dir)
    predictions = nn.predict(pred_input_fn)
    pred_array_holder = {}
    cur = next(predictions)
    for key in cur:
        if cur[key] is not None:
            pred_array_holder[key] = np.full([n] + list(cur[key].shape), -1.0, dtype=np.float32)
            pred_array_holder[key][0] = cur[key]
    # todo, this wraps, even when new predictions run out, fix, here and Opti!
    for i in range(1, n, 1):
        cur = next(predictions)
        for key in pred_array_holder:
            pred_array_holder[key][i] = cur[key]

    # labels is None in predict mode, so they have to be acquired separately
    graph2 = tf.Graph()
    with graph2.as_default():
        sess2 = tf.Session()
        x, gt = gt_input_fn()
        ground_truth = gt.eval(session=sess2)

    pred_array_holder['ground_truth'] = ground_truth

    with open(fileout, 'wb') as f:
        pickle.dump(pred_array_holder, f)
    print('saved predictions as {}'.format(fileout))


def pred2array(predictions, keys, n=32):
    out = {}
    first = next(predictions)
    for key in keys:
        out[key] = np.full([n] + list(first[key].shape), -111)
        out[key][0] = first[key]

    for i in range(1, n, 1):
        current = next(predictions)
        for key in keys:
            out[key][i] = current[key]
    return out


def main_playground(data_dir, train_dir, model_fn, problem, n=32, fileout='playground.pkl'):
    first_input_fn = input_fn(n, tf.estimator.ModeKeys.EVAL, data_dir, problem)
    second_input_fn = input_fn(n, tf.estimator.ModeKeys.EVAL, data_dir, problem)
    model_params = {'labels_shape': problem.label_shape}
    nn = tf.estimator.Estimator(model_fn=model_fn, params=model_params, model_dir=train_dir)
    nn.train(input_fn=first_input_fn, steps=3)
    predictions = nn.predict(first_input_fn)
    pred0 = pred2array(predictions, ['predictions', 'features'])
    nn.train(input_fn=first_input_fn, steps=3)
    predictions = nn.predict(first_input_fn)
    pred1 = pred2array(predictions, ['predictions', 'features'])
    nn.train(input_fn=second_input_fn, steps=3)
    predictions = nn.predict(first_input_fn)
    pred2 = pred2array(predictions, ['predictions', 'features'])
    predictions = nn.predict(first_input_fn)
    pred3 = pred2array(predictions, ['predictions', 'features'])
    out = [pred0, pred1, pred2, pred3]

    with open(fileout, 'wb') as f:
        pickle.dump(out, f)
    print('saved playground as {}'.format(fileout))


def main_preview(data_dir, problem, train_dir=None, model_fn=None, weight_keys=None, activation_keys=None,
                 list_only=False, n=17, fileout=None):
    logging.info('starting preview')
    if fileout is None:
        if train_dir is not None:
            fileout = train_dir + '/previews.pkl'
        else:
            fileout = 'previews.pkl'
    out = {'activations': {},
           'weights': {},
           'input': {}}
    # open session (on graphics card, for running things)
    sess = tf.Session()
    # setup input for data (function & place holders)
    preview_input_fn = input_fn(n, tf.estimator.ModeKeys.PREDICT, data_dir, problem)
    x, y = preview_input_fn()
    # actually evaluate, so NN can be fed with numbers
    feed_dict = input_fn_2_feed_dict(x, y, sess)

    # get graph setup
    if train_dir is not None:
        # load graph from checkpount
        try:
            latest_checkpoint = get_latest_checkpoint(train_dir)
        except IndexError:
            raise FileNotFoundError('No checkpoint files ending with expected ".meta" found in {}'.format(train_dir))

        new_saver = tf.train.import_meta_graph(latest_checkpoint + '.meta')
        new_saver.restore(sess, latest_checkpoint)
    elif model_fn is not None:
        # on fail, initialize graph from model_fn
        model_fn(x, y, tf.estimator.ModeKeys.TRAIN, {'labels_shape': problem.label_shape})
        tf.global_variables_initializer().run(session=sess)
    else:
        print("error: either 'model_fn' or 'train_dir' must not be none to build and preview graph")
        sys.exit(1)

    # two modes available
    # either list names of everything that could be returned
    if list_only:
        # list of all weights
        for item in tf.trainable_variables():
            out['weights'][item.name] = 1
        # list of all activations
        all_operations = sess.graph.get_operations()
        for item in all_operations:
            for tensor in item.values():
                out['activations'][tensor.name] = 1

    # or we save actual values for specific requests
    else:
        if activation_keys is not None:
            graph = tf.get_default_graph()
            for key in activation_keys:
                # first get the tensor from the graph
                a_tensor = graph.get_tensor_by_name(key)
                # then feed real numbers in and evaluate to get the activations
                out['activations'][key] = a_tensor.eval(feed_dict=feed_dict, session=sess)

        if weight_keys is not None:
            for key in weight_keys:
                # get weights by name
                var = [v for v in tf.trainable_variables() if v.name == key][0]
                # the weights are input independent, so just evaluate
                out['weights'][key] = var.eval(session=sess)
        # a copy of the input values is nice as well
        out['input'] = feed_dict

    sess.close()

    # save the 'out' dictionary as a pickle object
    with open(fileout, 'wb') as f:
        pickle.dump(out, f)
        print('predictions written to {}'.format(fileout))


def main_train(data_dir, train_dir, model_fn, problem, times_total=10000, evaluate_every=1000, learning_rate=1e-5,
               batch_size=32, params=None):
    if params is None:
        params = {}
    logging.info('starting training')
    eval_with = 50
    number_rounds = times_total // evaluate_every

    train_input_fn = input_fn(batch_size, tf.estimator.ModeKeys.TRAIN, data_dir, problem)
    dev_input_fn = input_fn(batch_size, tf.estimator.ModeKeys.EVAL, data_dir, problem)

    params.update({'learning_rate': learning_rate, 'labels_shape': problem.label_shape})
    nn = tf.estimator.Estimator(model_fn=model_fn, params=params, model_dir=train_dir)

    for i in range(number_rounds):
        print('training {}'.format(i))
        nn.train(input_fn=train_input_fn, steps=evaluate_every)

        print('evaluating {}'.format(i))
        nn.evaluate(input_fn=train_input_fn, steps=eval_with, name='training')
        nn.evaluate(input_fn=dev_input_fn, steps=eval_with, name='xvalidation')
    exporter = hub.LatestModuleExporter("an_exporter", problem.serving_input_fn_mod())  # todo, what's the input_fn do here?
    exporter.export(nn, "exported_modules", nn.latest_checkpoint())


def main(data_dir, train_dir, prob_string, mode, model_string, list_only, weights, activations, fileout, total,
         evaluate_every, log_file, log_level, n_proc, learning_rate, batch_size, params):

    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % log_level)

    # setup logging
    logging.basicConfig(filename=log_file, level=numeric_level, format='%(asctime)s|%(levelname)s|%(message)s',
                        datefmt="%Y.%m.%d %H:%M:%S")

    problem = eval(prob_string + '()')
    try:  # not all problems necessarily have / use threads atm, particularly not like this :-(
        problem.num_generate_tasks = n_proc
    except NotImplementedError:
        pass
    except AttributeError:
        pass

    # todo, add params to all modes, not just training
    params = json.loads(params)

    if model_string is not None:
        model_fn = eval(model_string)
    else:
        model_fn = None

    if mode == 'datagen':
        main_datagen(data_dir, problem)
    elif mode == 'train':
        main_train(data_dir, train_dir, model_fn, problem, times_total=total, evaluate_every=evaluate_every,
                   learning_rate=learning_rate, batch_size=batch_size, params=params)
    elif mode == 'predict':
        main_predict(data_dir, train_dir, model_fn, problem)
    elif mode == 'preview':
        try:
            weight_keys = weights.split(',')
        except AttributeError:
            weight_keys = None
        try:
            activation_keys = activations.split(',')
        except AttributeError:
            activation_keys = None
        main_preview(data_dir, problem, train_dir, model_fn, list_only=list_only,
                     activation_keys=activation_keys, weight_keys=weight_keys, fileout=fileout)
    elif mode == 'play':
        main_playground(data_dir, train_dir, model_fn, problem)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    modeouter = parser.add_argument_group("mode, select one")
    flags = parser.add_argument_group('required')

    mode = modeouter.add_mutually_exclusive_group(required=True)
    mode.add_argument('--train', action='store_true')
    mode.add_argument('--generate_data', action='store_true')
    mode.add_argument('--predict', action='store_true')
    mode.add_argument('--preview', action='store_true')
    mode.add_argument('--play', action='store_true')

    flags.add_argument("--data_dir", nargs='?', required=True, type=str,
                       help="data directory (output for data gen., input for training)")
    flags.add_argument("--problem", nargs='?', required=True, type=str, help="ClassName of problem (think data desc.)")

    training = parser.add_argument_group('training arguments')
    training.add_argument('--train_dir', nargs='?', type=str, help='model storage directory', default=None)
    training.add_argument('--model', nargs='?', type=str, help='name of model_fn', default=None)
    training.add_argument('-t', '--total', default=1000000, type=int, help='total iterations (default 1000000)')
    training.add_argument('-e', '--evaluate_every', default=1000, type=int,
                          help='evaluate every _ iterations (default (1000)')
    training.add_argument('-l', '--learning_rate', default=1e-5, type=float, help='learning rate (default 1e-5)')
    training.add_argument('-b', '--batch_size', default=32, type=int, help='batch size (default 32)')

    predicting = parser.add_argument_group('preview arguments',
                                           'preview needs either --train_dir or --model (former takes precedence)')
    predicting.add_argument('--available', action='store_true', help='dump available names instead of values')
    predicting.add_argument('--activations', nargs='?', default=None, type=str,
                            help='comma seperated list of activations to preview')
    predicting.add_argument('--weights', nargs='?', default=None, type=str,
                            help='comma separated list of weights to preview')
    predicting.add_argument('--pickle', nargs='?', default=None, type=str,
                            help='custom path for output .pkl file, default: {train_dir|`pwd`}/preview.pkl')

    optional = parser.add_argument_group('optional')
    optional.add_argument('--params', default="{}", type=str,
                          help="""json string of additional parameters (useful for some models), e.g. '{"filter_depth": 32}'""")
    optional.add_argument('-p', '--processes', default=1, type=int, help='n processes to start for data_generation')
    optional.add_argument('--log_level', default='info')
    optional.add_argument('--log_file', default='run.log')
    optional.add_argument('-h', '--help', action='help')
    args = parser.parse_args()
    # resolve modes

    # todo, this is basically redundant with main's if/elif fun
    if args.train:
        run_mode = 'train'
    elif args.generate_data:
        run_mode = 'datagen'
    elif args.predict:
        run_mode = 'predict'
    elif args.preview:
        run_mode = 'preview'
    elif args.play:
        run_mode = 'play'
    else:
        raise ValueError('missing input mode')

    main(data_dir=args.data_dir, train_dir=args.train_dir, mode=run_mode, prob_string=args.problem,
         model_string=args.model, list_only=args.available, weights=args.weights, activations=args.activations,
         fileout=args.pickle, total=args.total, evaluate_every=args.evaluate_every, log_file=args.log_file,
         log_level=args.log_level, n_proc=args.processes, learning_rate=args.learning_rate, batch_size=args.batch_size,
         params=args.params)
