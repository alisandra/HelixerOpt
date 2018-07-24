__author__ = 'Alisandra Denton'

import tensorflow as tf
import numpy as np
from builtins import range


# main models
def get_dropout_prob(mode, train_prob=0.5):
    """prepare dropout probability for neurons to regularize only during training"""
    if mode == tf.estimator.ModeKeys.TRAIN:
        dropout_prob = train_prob
    else:
        dropout_prob = 1
    return dropout_prob


def batch_out_sizes(features, labels, mode, pred_batch_size=128, pred_out_size=128):
    """Take note of useful input / output dimensions for later use"""
    batch_size = features.get_shape()[0]
    try:
        output_size = labels.get_shape()[1:]
    except AttributeError:
        output_size = None
    if mode == tf.estimator.ModeKeys.PREDICT:

        batch_size = pred_batch_size
        if output_size is None:
            output_size = pred_out_size
    print('os: {}'.format(output_size))
    print('bs: {}'.format(batch_size))
    return batch_size, output_size


def std_dilation_1d(layer_in, depth=64, width=5, activation=tf.nn.leaky_relu, padding="SAME", dilation_rate=2):
    return tf.layers.conv1d(layer_in, depth, width, activation=activation, padding=padding, dilation_rate=dilation_rate)


def score_exp_n_train(pre_predictions, mode, labels, params=None):
    """score / train setup for threshold classifier"""
    # set learning rate
    if params is None:
        params = {}
    if 'learning_rate' in params:
        learning_rate = params['learning_rate']
    else:
        learning_rate = 1e-4

    # so, pre_predictions has been relu'd. Predictions needs log space
    threshed_labels = tf.cast(labels >= 10, tf.float32)
    predictions = tf.cast(pre_predictions >= 0, tf.float32)

    # In prediction mode, return predictions
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
          mode=mode,
          predictions={"predictions": pre_predictions})

    # Calculate loss
    loss = tf.losses.sigmoid_cross_entropy(threshed_labels, pre_predictions)

    # actual optimization functions
    optimizer = tf.train.AdamOptimizer(learning_rate, epsilon=1e-6)
    train_op = optimizer.minimize(
        loss=loss, global_step=tf.train.get_global_step())

    # Calculate something to help evaluating
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(threshed_labels, predictions),
        # float32 seems to be required for the pearson correlation
        "FN": tf.metrics.false_negatives(threshed_labels, predictions),
        "FP": tf.metrics.false_positives(threshed_labels, predictions)
    }
    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metric_ops)


def score_sigmoid_n_train(pre_predictions, mode, labels, to_predict=None, params=None):
    """score / train setup for (binary or multiclass) classifier"""
    # set learning rate
    learning_rate = params['learning_rate']
    if to_predict is None:
        to_predict = {}
    # so, pre_predictions has been relu'd. Predictions needs log space
    predictions = tf.cast(pre_predictions >= 0, tf.float32)

    # In prediction mode, return predictions
    if mode == tf.estimator.ModeKeys.PREDICT:
        to_predict['predictions'] = predictions
        return tf.estimator.EstimatorSpec(
          mode=mode,
          predictions=to_predict)

    # Calculate loss
    loss = tf.losses.sigmoid_cross_entropy(labels, pre_predictions)

    # actual optimization functions
    optimizer = tf.train.AdamOptimizer(learning_rate, epsilon=1e-6)
    train_op = optimizer.minimize(
        loss=loss, global_step=tf.train.get_global_step())

    # Calculate something to help evaluating
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(labels, predictions),
        # float32 seems to be required for the pearson correlation
        "FN": tf.metrics.false_negatives(labels, predictions),
        "FP": tf.metrics.false_positives(labels, predictions)
    }
    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metric_ops)


def score_not_poisson(pre_predictions, mode, labels, params=None):
    """score maybe finally poisson"""
    # set learning rate
    if params is None:
        params = {}
    if 'learning_rate' in params:
        learning_rate = params['learning_rate']
    else:
        learning_rate = 1e-4

    # so, pre_predictions has been relu'd. Predictions needs log space
    predictions = tf.cast(pre_predictions >= 0, tf.float32)

    # In prediction mode, return predictions
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
          mode=mode,
          predictions={"predictions": pre_predictions})

    # Calculate loss
    # loss = tf.nn.log_poisson_loss(tf.reshape(labels, [-1, np.prod(labels.get_shape()[1:])]),
    #                               tf.reshape(pre_predictions, [-1, np.prod(pre_predictions.get_shape()[1:])]))
    log_lab = tf.log1p(labels)
    log_pred = tf.log1p(tf.nn.relu(pre_predictions))
    loss = tf.losses.mean_squared_error(log_lab, log_pred)
    # actual optimization functions
    optimizer = tf.train.AdamOptimizer(learning_rate, epsilon=1e-6)
    train_op = optimizer.minimize(
        loss=loss, global_step=tf.train.get_global_step())

    # Calculate something to help evaluating
    eval_metric_ops = {
        "RMSE": tf.metrics.root_mean_squared_error(labels, pre_predictions),
        "RMSE logspace": tf.metrics.root_mean_squared_error(log_lab, log_pred),
        "Pearson's r": tf.contrib.metrics.streaming_pearson_correlation(log_lab, log_pred)
        # float32 seems to be required for the pearson correlation
    }
    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metric_ops)


def still_not_poisson(pre_predictions, mode, labels, params=None):
    """score maybe finally poisson"""
    # set learning rate
    if params is None:
        params = {}
    if 'learning_rate' in params:
        learning_rate = params['learning_rate']
    else:
        learning_rate = 1e-4

    # Calculate loss
    # loss = tf.nn.log_poisson_loss(tf.reshape(labels, [-1, np.prod(labels.get_shape()[1:])]),
    #                               tf.reshape(pre_predictions, [-1, np.prod(pre_predictions.get_shape()[1:])]))

    log_pred = tf.nn.relu(pre_predictions)  # should push the predictions towards being in log space this way
    # In prediction mode, return predictions
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
          mode=mode,
          predictions={"predictions": log_pred})
    log_lab = tf.log1p(labels)
    loss = tf.losses.mean_squared_error(log_lab, log_pred)
    # actual optimization functions
    optimizer = tf.train.AdamOptimizer(learning_rate, epsilon=1e-6)
    train_op = optimizer.minimize(
        loss=loss, global_step=tf.train.get_global_step())

    # Calculate something to help evaluating
    eval_metric_ops = {
        "RMSE logspace": tf.metrics.root_mean_squared_error(log_lab, log_pred),
        # float32 seems to be required for the pearson correlation
        "Pearson's r": tf.contrib.metrics.streaming_pearson_correlation(log_lab, log_pred)
    }
    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metric_ops)


def score_poisson(pre_predictions, mode, labels, params=None):
    """score maybe finally poisson"""
    # set learning rate
    if params is None:
        params = {}
    if 'learning_rate' in params:
        learning_rate = params['learning_rate']
    else:
        learning_rate = 1e-4

    # so, pre_predictions has been relu'd. Predictions needs log space
    predictions = tf.cast(pre_predictions >= 0, tf.float32)

    # In prediction mode, return predictions
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
          mode=mode,
          predictions={"predictions": pre_predictions})

    # Calculate loss
    # loss = tf.nn.log_poisson_loss(tf.reshape(labels, [-1, np.prod(labels.get_shape()[1:])]),
    #                               tf.reshape(pre_predictions, [-1, np.prod(pre_predictions.get_shape()[1:])]))
    log_lab = tf.log1p(labels)
    log_pred = tf.log1p(tf.nn.relu(pre_predictions))
    loss = tf.nn.log_poisson_loss(log_lab, log_pred)
    # actual optimization functions
    optimizer = tf.train.AdamOptimizer(learning_rate, epsilon=1e-6)
    train_op = optimizer.minimize(
        loss=loss, global_step=tf.train.get_global_step())

    # Calculate something to help evaluating
    eval_metric_ops = {
        "RMSE": tf.metrics.root_mean_squared_error(labels, pre_predictions),
        "RMSE logspace": tf.metrics.root_mean_squared_error(log_lab, log_pred)
        # float32 seems to be required for the pearson correlation
    }
    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metric_ops)


def conv_and_dil(u3_hidden_reshape, keep_prob=0.9):
    u2_hidden_layer = tf.layers.conv1d(u3_hidden_reshape, 64, 5, activation=tf.nn.relu, padding="SAME")
    u2_hidden_pool = tf.layers.max_pooling1d(u2_hidden_layer, strides=2, pool_size=2, padding='valid')
    # 2
    u1_hidden_layer = tf.layers.conv1d(u2_hidden_pool, 64, 5, activation=tf.nn.relu, padding="SAME")
    u1_hidden_pool = tf.layers.max_pooling1d(u1_hidden_layer, strides=2, pool_size=2, padding='valid')
    # 2
    u1b_hidden_layer = tf.layers.conv1d(u1_hidden_pool, 64, 5, activation=tf.nn.relu, padding="SAME")
    u1b_hidden_pool = tf.layers.max_pooling1d(u1b_hidden_layer, strides=2, pool_size=2, padding='valid')
    # 2
    u1c_hidden_layer = tf.layers.conv1d(u1b_hidden_pool, 64, 5, activation=tf.nn.relu, padding="SAME")
    u1c_hidden_pool = tf.layers.max_pooling1d(u1c_hidden_layer, strides=2, pool_size=2, padding='valid')
    # trying to add dropout to convolutional layers (drop whole filters)
    curr_shape = tf.cast(tf.shape(u1c_hidden_pool), tf.int32)
    batch_size = curr_shape[0]
    mid = curr_shape[1]
    conv_drop = tf.nn.dropout(u1c_hidden_pool, keep_prob=keep_prob, noise_shape=[batch_size, mid, 1])
    # 3 (start dilation)
    oo_hidden_layer = std_dilation_1d(conv_drop)
    # 4
    first_hidden_layer = std_dilation_1d(oo_hidden_layer)
    # 5
    second_hidden_layer = std_dilation_1d(first_hidden_layer)
    third_hidden_layer = std_dilation_1d(second_hidden_layer)
    # another round of dropout
    curr_shape01 = tf.cast(tf.shape(third_hidden_layer), tf.int32)
    batch_size01 = curr_shape01[0]
    mid01 = curr_shape01[1]
    dil_conv_drop = tf.nn.dropout(third_hidden_layer, keep_prob=keep_prob, noise_shape=[batch_size01, mid01, 1])

    fourth_hidden_layer = std_dilation_1d(dil_conv_drop)
    fifth_hidden_layer = std_dilation_1d(fourth_hidden_layer)
    return fifth_hidden_layer


def dilation05_model_fn(features, labels, mode, params):
    """Model function for Estimator - Somewhat trying to copy Basset paper, 11 conv, 6 dilation"""

    # setup useful parameters
    batch_size, output_size = batch_out_sizes(features, labels, mode)
    dropout_prob = get_dropout_prob(mode, 0.7)

    # prep input
    x_image = features
    print(x_image.get_shape())
    print(labels.get_shape())

    # convolutional layer 1
    u3_hidden_layer = tf.layers.conv1d(x_image, 32, 5, activation=tf.nn.relu, padding="valid")
    # drop flat dimension
    print('u3_hidden_layer shape: {}'.format(u3_hidden_layer.get_shape()))
    u3_hidden_reshape = tf.reshape(u3_hidden_layer, (-1, u3_hidden_layer.get_shape()[1], 32))

    fifth_hidden_layer = conv_and_dil(u3_hidden_reshape)
    # in theory we have some width now
    print("fifth_hidden shape: {}".format(fifth_hidden_layer.get_shape()))
    # flatten
    last_conv_flat = tf.reshape(fifth_hidden_layer, [-1, np.prod(fifth_hidden_layer.get_shape()[1:])])
    print("flat shape: {}".format((last_conv_flat.get_shape())))
    last_conv_drop = tf.layers.dropout(last_conv_flat, dropout_prob)
    print("last_conv_drop shape: {}".format(last_conv_drop.get_shape()))
    # still use relu below because I will need to take the log
    flat_predictions = tf.layers.dense(last_conv_drop, np.prod(output_size), activation=None)
    pre_predictions = tf.reshape(flat_predictions, [-1] + list(output_size))

    estimator_spec = score_exp_n_train(pre_predictions, mode, labels)
    return estimator_spec


def meety_bits_dilation(features, labels, mode, params):
    # setup useful parameters
    #batch_size, output_size = batch_out_sizes(features, labels, mode) #todo, clean up
    output_size = params['labels_shape']
    dropout_prob = get_dropout_prob(mode, 0.2)
    dropout_conv_prob = get_dropout_prob(mode, 0.7)

    # prep input
    x_image = features
    print(x_image.get_shape())
    #print(labels.get_shape())

    # convolutional layer 1
    u3_hidden_layer = tf.layers.conv1d(x_image, 32, 5, activation=tf.nn.relu, padding="valid")
    # drop flat dimension
    print('u3_hidden_layer shape: {}'.format(u3_hidden_layer.get_shape()))
    u3_hidden_reshape = tf.reshape(u3_hidden_layer, (-1, u3_hidden_layer.get_shape()[1], 32))

    fifth_hidden_layer = conv_and_dil(u3_hidden_reshape)
    # in theory we have some width now
    print("fifth_hidden shape: {}".format(fifth_hidden_layer.get_shape()))
    # flatten
    last_conv_flat = tf.reshape(fifth_hidden_layer, [-1, np.prod(fifth_hidden_layer.get_shape()[1:])])
    print("flat shape: {}".format((last_conv_flat.get_shape())))
    last_conv_drop = tf.layers.dropout(last_conv_flat, dropout_prob)
    print("last_conv_drop shape: {}".format(last_conv_drop.get_shape()))
    # still use relu below because I will need to take the log
    flat_predictions = tf.layers.dense(last_conv_drop, np.prod(output_size), activation=None)
    pre_predictions = tf.reshape(flat_predictions, [-1] + list(output_size))
    return pre_predictions


def dilation_pos00_model_fn(features, labels, mode, params):
    """Model function for Estimator - Somewhat trying to copy Basset paper, 11 conv, 6 dilation"""
    pre_predictions = meety_bits_dilation(features, labels, mode, params)
    estimator_spec = score_not_poisson(pre_predictions, mode, labels, params)
    return estimator_spec


def dilation_pos01_model_fn(features, labels, mode, params):
    pre_predictions = meety_bits_dilation(features, labels, mode, params)
    estimator_spec = still_not_poisson(pre_predictions, mode, labels, params)
    return estimator_spec


def dilation_pos02_model_fn(features, labels, mode, params):
    """Checked while building dilation CNN"""
    output_size = params['labels_shape']
    print(output_size)
    dropout_prob = get_dropout_prob(mode, 0.9)

    features_shape = features.get_shape()
    print(features_shape)
    x_image = features#tf.reshape(features, (-1, features_shape[1], 4))
    # regular convolution
    u3_hidden_layer = tf.layers.conv1d(x_image, 32, 5, activation=tf.nn.relu, padding="same", name='popcorn')
    u3_hidden_pool = tf.layers.max_pooling1d(u3_hidden_layer, strides=2, pool_size=2, padding='valid')
    #u2_hidden
    u2_hidden_layer = tf.layers.conv1d(u3_hidden_pool, 64, 5, activation=tf.nn.relu, padding="SAME", name='coke')
    u2_hidden_pool = tf.layers.max_pooling1d(u2_hidden_layer, strides=2, pool_size=2, padding='valid')
    # 2
    u1_hidden_layer = tf.layers.conv1d(u2_hidden_pool, 128, 5, activation=tf.nn.relu, padding="SAME", name='soup')
    u1_hidden_pool = tf.layers.max_pooling1d(u1_hidden_layer, strides=2, pool_size=2, padding='valid')
    # reduce filter number
    u1_condense = tf.layers.conv1d(u1_hidden_pool, 64, 1, activation=tf.nn.relu, padding='same', name='sweet')

    # dilation
    oo_hidden_layer = tf.layers.conv1d(u1_condense, 128, 5, dilation_rate=2, padding='same', activation=tf.nn.relu)
    o1_hidden_layer = tf.layers.conv1d(oo_hidden_layer, 256, 5, dilation_rate=2, padding='same', activation=tf.nn.relu)
    o2_hidden_layer = tf.layers.conv1d(o1_hidden_layer, 512, 5, dilation_rate=2, padding='same', activation=tf.nn.relu)
    o2_condense = tf.layers.conv1d(o2_hidden_layer, 128, 1, activation=tf.nn.relu, padding='same', name='potato')
    o2_condenser = tf.layers.conv1d(o2_condense, 64, 1, activation=tf.nn.relu, padding='same')
    o2_shape = o2_condenser.get_shape()
    last_conv_flat = tf.reshape(o2_condenser, [-1, o2_shape[1] * o2_shape[2]], name='salmon')
    last_conv_drop = tf.layers.dropout(last_conv_flat, dropout_prob, name='mic')
    #meh = tf.reshape(last_conv_drop, [32, 32768])  # maybe go back to trying get_shape, as this
    # doesn't seem deterministic
    flat_predictions = tf.layers.dense(last_conv_drop, 3840, activation=None, name='toast')
    pre_predictions = tf.reshape(flat_predictions, [-1] + list(output_size), name='lemon')
    estimator_spec = still_not_poisson(pre_predictions, mode, labels, params)
    return estimator_spec


def dilation_basenji_model_fn(features, labels, mode, params):
    """Checked while building dilation CNN"""
    output_size = params['labels_shape']
    print(output_size)
    dropout_prob = get_dropout_prob(mode, 0.8)

    #features_shape = features.get_shape()
    dna_in = features#tf.reshape(features, (-1, features_shape[1], 4))

    # regular convolution
    filter_depth = 256
    kernel_size = 20
    pool_strides = [2, 2, 2, 4, 3, 3, 4]

    # normal convolutional layers
    conv_out = dna_in
    for i in range(4):
        convoluted = tf.layers.conv1d(conv_out, filter_depth, kernel_size, activation=tf.nn.relu, padding='same')
        pooled = tf.layers.max_pooling1d(convoluted, pool_size=pool_strides[i], strides=pool_strides[i])
        conv_out = tf.layers.dropout(pooled, rate=dropout_prob)

    # dilated convolutional layers
    dil_out = dna_in
    for i in range(7):
        dilated = tf.layers.conv1d(dil_out, filter_depth, kernel_size, activation=tf.nn.relu, padding='same',
                                   dilation_rate=2**(i + 1))
        # Basenji technically doesn't pool during dilation, but have to do something until I figure out fc
        pooled = tf.layers.max_pooling1d(dilated, pool_size=pool_strides[i], strides=pool_strides[i])
        dil_out = tf.layers.dropout(pooled, rate=dropout_prob)

    # concatenated
    hidden_concat = tf.concat([conv_out, dil_out], axis=1)  # AKA, filter depth

    # fully connected
    fully_connected = tf.layers.dense(hidden_concat, 256)
    full_norm = tf.contrib.layers.layer_norm(fully_connected)
    full_relu = tf.nn.relu(full_norm)
    full_dropout = tf.layers.dropout(full_relu, dropout_prob, name='soda')
    pennultimate_shape = full_dropout.get_shape()
    pennultimate_shape.assert_has_rank(3)
    print(pennultimate_shape)
    out = tf.expand_dims(full_dropout, 2, name='fanta')

    # and dense to output shape
    flat_dropout = tf.reshape(out, [-1, pennultimate_shape[1] * pennultimate_shape[2]], name='mountaindew')
    flat_predictions = tf.layers.dense(flat_dropout, np.prod(output_size), name='cola')
    pre_predictions = tf.reshape(flat_predictions, [-1] + output_size, name='sprite')

    estimator_spec = still_not_poisson(pre_predictions, mode, labels, params)
    return estimator_spec


# begin what was previously in more_models
def standardize_params(params):
    # set learning rate
    if params is None:
        params = {}
    if 'learning_rate' not in params:
        params['learning_rate'] = 1e-5
    return params


def conv1ds_w_pool(features, n_layers=5, filter_depth=256, kernel_size=9, pool_strides=0, dropout_prob=0.8,
                   dilation=1):
    to_predict = {}  # add tensors to this dictionary, if they should be previewable via predict
    # normal convolutional layers
    conv_out = features
    for i in range(n_layers):
        conv_out = tf.layers.conv1d(conv_out, filter_depth, kernel_size, activation=tf.nn.relu, padding='same',
                                    dilation_rate=dilation)
        if pool_strides:
            conv_out = tf.layers.max_pooling1d(conv_out, pool_size=pool_strides, strides=pool_strides)
        conv_out = tf.layers.batch_normalization(conv_out)
        conv_out = tf.layers.dropout(conv_out, rate=dropout_prob)
        print(conv_out.get_shape())
    return conv_out, to_predict


def conv2pre_predictions(conv_out, params, dropout_prob=0.8, filter_depth=256):
    to_predict = {}  # add tensors to this dictionary, if they should be previewable via predict
    output_size = params['labels_shape']
    fully_connected = tf.layers.dense(conv_out, filter_depth)
    print(fully_connected.get_shape())
    drop1 = tf.layers.dropout(fully_connected, rate=dropout_prob)

    pennultimate_shape = drop1.get_shape()
    pennultimate_shape.assert_has_rank(3)

    flat_dropout = tf.reshape(drop1, [-1, pennultimate_shape[1] * pennultimate_shape[2]])
    print(flat_dropout.get_shape())
    flat_predictions = tf.layers.dense(flat_dropout, np.prod(output_size))
    print(flat_predictions.get_shape())
    pre_predictions = tf.reshape(flat_predictions, [-1] + output_size, name='raw_predictions')

    return pre_predictions, to_predict


def score_n_spec_highlow(pre_predictions, mode, labels, to_predict=None, threshold=10, params=None):
    """score as low or high expression"""
    if to_predict is None:
        to_predict = {}

    # In prediction mode, return predictions
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = tf.sigmoid(pre_predictions)
        print('--> predictions: {}'.format(predictions))
        to_predict['predictions'] = predictions
        to_predict['pre_predictions'] = pre_predictions
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=to_predict)

    # set learning rate
    learning_rate = params['learning_rate']
    # convert labels into high_low
    high_low = tf.cast(labels >= threshold, tf.float32)

    loss = tf.losses.sigmoid_cross_entropy(logits=pre_predictions, multi_class_labels=high_low)
    # actual optimization functions
    optimizer = tf.train.AdamOptimizer(learning_rate, epsilon=1e-6)
    train_op = optimizer.minimize(
        loss=loss, global_step=tf.train.get_global_step())

    # predictions as 0, 1s
    final_pred = tf.cast(pre_predictions >= 0, tf.float32)
    # tensorflows builtin evaluation metrics
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(predictions=final_pred, labels=high_low),
        "FP": tf.metrics.false_positives(predictions=final_pred, labels=high_low),
        "FN": tf.metrics.false_negatives(predictions=final_pred, labels=high_low)
    }

    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metric_ops)


def score_n_spec_rmse(pre_predictions, mode, labels, to_predict=None, params=None):
    """score RMSE for expression, labels should already be log(y + 1) transformed and mean-centered"""
    if to_predict is None:
        to_predict = {}
    # convert labels into
    # In prediction mode, return predictions
    if mode == tf.estimator.ModeKeys.PREDICT:
        to_predict['predictions'] = pre_predictions
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=to_predict)

    loss = tf.losses.mean_squared_error(labels=labels, predictions=pre_predictions)
    # actual optimization functions
    optimizer = tf.train.AdamOptimizer(params['learning_rate'], epsilon=1e-6)
    train_op = optimizer.minimize(
        loss=loss, global_step=tf.train.get_global_step())

    # tensorflows builtin evaluation metrics
    eval_metric_ops = {
        "RMSE logspace": tf.metrics.root_mean_squared_error(labels, pre_predictions),
        # float32 seems to be required for the pearson correlation
        "Pearson's r": tf.contrib.metrics.streaming_pearson_correlation(labels, pre_predictions)
    }

    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metric_ops)


def score_n_spec_sigmoid(pre_predictions, mode, labels, to_predict=None, params=None):
    """standard labels should be 0,1... pre_predictions are logits"""
    if to_predict is None:
        to_predict = {}
    # convert labels into
    # In prediction mode, return predictions
    if mode == tf.estimator.ModeKeys.PREDICT:
        to_predict['predictions'] = tf.sigmoid(pre_predictions)
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=to_predict)

    loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=labels, logits=pre_predictions)
    # actual optimization functions
    optimizer = tf.train.AdamOptimizer(params['learning_rate'], epsilon=1e-6)
    train_op = optimizer.minimize(
        loss=loss, global_step=tf.train.get_global_step())

    # predictions as 0, 1s
    final_pred = tf.cast(pre_predictions >= 0, tf.float32)
    # tensorflows builtin evaluation metrics
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(predictions=final_pred, labels=labels),
        "FP": tf.metrics.false_positives(predictions=final_pred, labels=labels),
        "FN": tf.metrics.false_negatives(predictions=final_pred, labels=labels)
    }
    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metric_ops)


def score_n_spec_log_poisson(pre_predictions, mode, labels, to_predict=None, params=None):
    """log poisson loss, probably has to have batch size 1"""
    if to_predict is None:
        to_predict = {}
    # convert labels into
    # In prediction mode, return predictions
    if mode == tf.estimator.ModeKeys.PREDICT:
        to_predict['predictions'] = pre_predictions
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=to_predict)

    # todo, I'd like to check, but fails, when same??
    #assert list(labels.get_shape()) == list(pre_predictions.get_shape())
    flat_labels = tf.reshape(labels, [-1])
    flat_preds = tf.reshape(pre_predictions, [-1])

    # todo, figure out how log poisson loss is calculated, compare to code, and make sure log_input is right!
    loss = tf.nn.log_poisson_loss(targets=flat_labels, log_input=flat_preds)
    # actual optimization functions
    optimizer = tf.train.AdamOptimizer(params['learning_rate'], epsilon=1e-6)
    train_op = optimizer.minimize(
        loss=loss, global_step=tf.train.get_global_step())

    # tensorflows builtin evaluation metrics
    eval_metric_ops = {
        "RMSE logspace": tf.metrics.root_mean_squared_error(labels, pre_predictions),
        # float32 seems to be required for the pearson correlation
        "Pearson's r": tf.contrib.metrics.streaming_pearson_correlation(labels, pre_predictions)
    }

    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metric_ops)


def highlow_model_fn_01(features, labels, mode, params=None):
    """classifier scoring for tissue problems"""
    params = standardize_params(params)
    to_predict = {'features': features}
    dropout_prob = get_dropout_prob(mode, 0.8)
    # CNN
    conv_out, new_preds = conv1ds_w_pool(features, dropout_prob=dropout_prob, n_layers=3, filter_depth=128,
                                         pool_strides=2)
    to_predict.update(new_preds)
    conv_out, new_preds = conv1ds_w_pool(conv_out, dropout_prob=dropout_prob, n_layers=5, filter_depth=128,
                                         pool_strides=0, dilation=2)
    to_predict.update(new_preds)
    # convolutional output -> dense layer -> predictions shape
    pre_predictions, fc_preds = conv2pre_predictions(conv_out, params, dropout_prob=dropout_prob, filter_depth=128)
    to_predict.update(new_preds)
    # predictions, metrics, scoring, loss, optimization
    estimator_spec = score_n_spec_highlow(pre_predictions, mode, labels, params=params, to_predict=to_predict,
                                          threshold=1)
    return estimator_spec


def rmse_model_fn_01(features, labels, mode, params=None):
    """rmse scoring for tissue problems"""
    params = standardize_params(params)
    to_predict = {'features': features}
    dropout_prob = get_dropout_prob(mode, 0.8)
    # CNN
    conv_out, new_preds = conv1ds_w_pool(features, dropout_prob=dropout_prob, n_layers=3, filter_depth=128,
                                         pool_strides=2)
    to_predict.update(new_preds)
    conv_out, new_preds = conv1ds_w_pool(conv_out, dropout_prob=dropout_prob, n_layers=5, filter_depth=128,
                                         pool_strides=0, dilation=2)
    to_predict.update(new_preds)
    # convolutional output -> dense layer -> predictions shape
    pre_predictions, fc_preds = conv2pre_predictions(conv_out, params, dropout_prob=dropout_prob, filter_depth=128)
    to_predict.update(new_preds)
    # predictions, metrics, scoring, loss, optimization
    estimator_spec = score_n_spec_rmse(pre_predictions, mode, labels, params=params, to_predict=to_predict)
    return estimator_spec


def plain_cnn_model_fn_01(features, labels, mode, params=None):
    """as rmse_model_fn_01, except sigmoid cross entropy scoring for gene_calling"""
    params = standardize_params(params)
    to_predict = {'features': features}
    dropout_prob = get_dropout_prob(mode, 0.8)
    # CNN
    conv_out, new_preds = conv1ds_w_pool(features, dropout_prob=dropout_prob, n_layers=3, filter_depth=128,
                                         pool_strides=2)
    to_predict.update(new_preds)
    conv_out, new_preds = conv1ds_w_pool(conv_out, dropout_prob=dropout_prob, n_layers=5, filter_depth=128,
                                         pool_strides=0, dilation=2)
    to_predict.update(new_preds)
    # convolutional output -> dense layer -> predictions shape
    pre_predictions, fc_preds = conv2pre_predictions(conv_out, params, dropout_prob=dropout_prob, filter_depth=128)
    to_predict.update(new_preds)
    # predictions, metrics, scoring, loss, optimization
    estimator_spec = score_n_spec_sigmoid(pre_predictions, mode, labels, params=params, to_predict=to_predict)
    return estimator_spec


def res_block(inputs, depth=2, kernal_size=5, dilation=1, filter_depth=64, strides=1):
    cur = inputs
    cur = tf.layers.conv1d(cur, filter_depth, kernal_size, dilation_rate=dilation, padding='same', strides=strides)
    for _ in range(1, depth, 1):
        cur = tf.layers.conv1d(cur, filter_depth, kernal_size, dilation_rate=dilation, padding='same', strides=1)
        cur = tf.layers.batch_normalization(cur)
        cur = tf.nn.relu(cur)
    input_shape = list(inputs.get_shape())
    # if any dimensions don't match, perform a 1 by convolution so that they do
    if (input_shape[2] != filter_depth) | (strides != 1):
        inputs = tf.layers.conv1d(inputs, filter_depth, 1, strides=strides)

    out = tf.add(cur, inputs)
    return out


def best_guess_model_fn_01(features, labels, mode, params=None):
    to_predict = {'features': features}
    pre_predictions, to_predict = resnet_down_by_32_n_crop(features, labels, to_predict, params)

    pre_predictions = tf.nn.relu(pre_predictions)  # positive values only
    #return score_n_spec_log_poisson(pre_predictions, mode, labels, to_predict=to_predict, params=params)
    return score_n_spec_rmse(pre_predictions, mode, labels, to_predict=to_predict, params=params)


def best_guess_model_fn_02(features, labels, mode, params=None):
    to_predict = {'features': features}
    pre_predictions, to_predict = resnet_down_by_32_n_crop(features, labels, to_predict, params)
    return score_n_spec_highlow(pre_predictions, mode, labels, to_predict=to_predict, params=params, threshold=10)


def best_guess_model_fn_03(features, labels, mode, params=None):
    features = tf.cast(features, tf.float16)
    if labels is not None:
         labels = tf.cast(labels, tf.float16)
    to_predict = {'features': features}
    pre_predictions, to_predict = resnet_down_by_128_n_fc(features, labels, to_predict, params)
    return score_n_spec_highlow(pre_predictions, mode, labels, to_predict=to_predict, params=params, threshold=10)


def resnet_down_by_32_n_crop(features, labels, to_predict, params=None):

    # stack up several convolutional layers
    cur = features

    cur = res_block(cur, filter_depth=64)
    cur = res_block(cur, filter_depth=128)
    dil = res_block(cur, dilation=2, depth=2, filter_depth=128)
    dil = res_block(dil, dilation=2, depth=3, filter_depth=128)
    dil = res_block(dil, dilation=2, depth=3, filter_depth=256)
    dil = res_block(dil, dilation=2, depth=3, filter_depth=256)
    dil = tf.layers.conv1d(dil, 256, 1, strides=8)  # or max pool?

    notdil = res_block(cur, strides=2, filter_depth=128)
    notdil = res_block(notdil, filter_depth=128)
    notdil = res_block(notdil, strides=2, filter_depth=256)
    notdil = res_block(notdil, filter_depth=256)
    notdil = res_block(notdil, strides=2, filter_depth=256)

    skip = tf.layers.conv1d(cur, 128, 1, strides=8)
    merge_in = tf.concat([dil, notdil, skip], axis=2)
    # keep adding simple res blocks until we've halved it 5 times (2 done so far)
    almost = res_block(merge_in, filter_depth=512, strides=2)
    almost = res_block(almost, filter_depth=512, strides=2)
    #almost = res_block(almost, filter_depth=512, strides=2)
    labels_shape = params['labels_shape']

    # we will only take the middle half
    almost_shape = list(almost.get_shape())
    assert almost_shape[1] == labels_shape[0] * 2
    start = almost_shape[1] // 4
    stop = start * 3
    final_depth = 17
    padd = final_depth // 2
    cropped = almost[:, (start - padd):(stop + padd), :]

    pre_predictions = tf.layers.conv1d(cropped, labels_shape[1], final_depth, padding='valid')
    print('pre_predictions {}'.format(pre_predictions.get_shape()))
    print('labels {}'.format(labels_shape))
    return pre_predictions, to_predict


def resnet_down_by_128_n_fc(features, labels, to_predict, params=None):

    # stack up several convolutional layers
    cur = features

    cur = res_block(cur, filter_depth=64)
    cur = res_block(cur, filter_depth=128)
    dil = res_block(cur, dilation=2, depth=2, filter_depth=128)
    dil = res_block(dil, dilation=2, depth=3, filter_depth=128)
    dil = res_block(dil, dilation=2, depth=3, filter_depth=256)
    dil = res_block(dil, dilation=2, depth=3, filter_depth=256)
    dil = tf.layers.conv1d(dil, 256, 1, strides=8)  # or max pool?

    notdil = res_block(cur, strides=2, filter_depth=128)
    notdil = res_block(notdil, filter_depth=128)
    notdil = res_block(notdil, strides=2, filter_depth=256)
    notdil = res_block(notdil, filter_depth=256)
    notdil = res_block(notdil, strides=2, filter_depth=256)

    skip = tf.layers.conv1d(cur, 128, 1, strides=8)
    merge_in = tf.concat([dil, notdil, skip], axis=2)
    # keep adding simple res blocks until we've halved it 5 times (2 done so far)
    almost = res_block(merge_in, filter_depth=512, strides=2)
    almost = res_block(almost, filter_depth=512, strides=2)
    almost = res_block(almost, filter_depth=512, strides=2)
    almost = res_block(almost, filter_depth=512, strides=2)
    #almost = res_block(almost, filter_depth=512, strides=2)
    labels_shape = params['labels_shape']

    # we will only take the middle half
    almost_shape = np.array(almost.get_shape())
    print('almost shape {}'.format(almost_shape))
    print(np.prod(almost_shape[1:]))
    almost = tf.reshape(almost, [-1] + [np.prod(almost_shape[1:])])
    print('almost get shape {}'.format(almost.get_shape()))

    pre_predictions = tf.layers.dense(almost, np.prod(labels_shape))
    print('tmp pre_predictions get shape {}'.format(pre_predictions.get_shape()))
    pre_predictions = tf.reshape(pre_predictions, [-1] + labels_shape)
    print('pre_predictions {}'.format(pre_predictions.get_shape()))
    print('labels {}'.format(labels_shape))

    to_predict['cur'] = cur
    to_predict['dil'] = dil
    to_predict['skip'] = skip
    to_predict['merge_in'] = merge_in
    to_predict['almost'] = almost
    to_predict['pre_predictions'] = pre_predictions
    return pre_predictions, to_predict


# auto encoders
def autoencode_fs_01(features, labels, mode, params=None):
    """convolutional in, dense out autoencoder for fieldspec data"""
    params = standardize_params(params)
    to_predict = {'features': features}
    dropout_prob = get_dropout_prob(mode, 0.8)
    # CNN
    conv_out, new_preds = conv1ds_w_pool(features, dropout_prob=dropout_prob, n_layers=4, filter_depth=4,
                                         pool_strides=2)
    # should deconvolute back up, but todo
    conv_out_shape = np.array(conv_out.get_shape())
    print('conv_out_shape: {}'.format(conv_out_shape))
    flattened = tf.reshape(conv_out, [-1] + [np.prod(conv_out_shape[1:])])
    pre_predictions = tf.layers.dense(flattened, np.prod(params['labels_shape']))
    pre_predictions = tf.reshape(pre_predictions, [-1] + params['labels_shape'])
    # predictions, metrics, scoring, loss, optimization
    estimator_spec = score_n_spec_rmse(pre_predictions, mode, labels=features, params=params, to_predict=to_predict)
    return estimator_spec


def mk_convolv_fn(n_layers=8, filter_depth=4):
    # making this as a function, bc the example does it, and bc it somehow makes sense that things like
    # n_layers are set for a module being imported/exported
    # todo, make official inputs/outputs, signature
    def model_fn(features, dropout_prob):
        conv_out, new_preds = conv1ds_w_pool(features, dropout_prob=dropout_prob, n_layers=n_layers,
                                             filter_depth=filter_depth, pool_strides=2)
        return conv_out, new_preds

    return model_fn


def autoencode_fs_02(features, labels, mode, params=None):
    """convolutional in, part convolutional, part dense autoencoder for fieldspec data"""
    params = standardize_params(params)
    to_predict = {'features': features}
    try:
        n_conv_layers = params['n_convolutions']
    except KeyError:
        n_conv_layers = 8

    try:
        filter_depth = params['filter_depth']
    except KeyError:
        filter_depth = 4

    dropout_prob = get_dropout_prob(mode, 0.8)
    # CNN
    #conv_fn = mk_convolv_fn(n_layers=n_conv_layers, filter_depth=filter_depth)
    # todo, replace with module instance
    # todo, figure out how to get module instance back out (e.g. similar to predict method...)
    conv_out, new_preds = conv1ds_w_pool(features, dropout_prob=dropout_prob, n_layers=n_conv_layers,
                                         filter_depth=filter_depth, pool_strides=2)

    conv_out_shape = np.array(conv_out.get_shape())
    print('conv_out_shape: {}'.format(conv_out_shape))
    # a bit hackish for de-convolution, as no conv1d_transpose was available.
    deconv_out = tf.reshape(conv_out, [-1] + list(conv_out_shape[1:]) + [1])
    for _ in range(n_conv_layers - 5):
        deconv_out = tf.layers.conv2d_transpose(deconv_out, filters=filter_depth, kernel_size=[9, 1], strides=[2, 1])
    deconv_out_shape = np.array(deconv_out.get_shape())
    print('deconv_out_shape: {}'.format(deconv_out_shape))
    flattened = tf.reshape(deconv_out, [-1] + [np.prod(deconv_out_shape[1:])])
    print('to flat or not to flat: {}'.format(flattened.get_shape()))
    pre_predictions = tf.layers.dense(flattened, np.prod(params['labels_shape']))
    print('pre_predictions shape{}'.format(pre_predictions.get_shape()))
    pre_predictions = tf.reshape(pre_predictions, [-1] + params['labels_shape'])
    # predictions, metrics, scoring, loss, optimization
    estimator_spec = score_n_spec_rmse(pre_predictions, mode, labels=features, params=params, to_predict=to_predict)
    return estimator_spec


def logistic_reg(features, labels, mode, params=None):
    """simple logistic regression"""
    params = standardize_params(params)
    lab_size = np.prod(params['labels_shape'])
    print(lab_size)
    pre_predictions = tf.layers.dense(features, lab_size, kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001))
    print(pre_predictions.get_shape())
    pre_predictions = tf.reshape(pre_predictions, [-1] + params['labels_shape'])
    print(pre_predictions.get_shape())
    to_predict = {'features': features, 'labels': labels, 'pre_predictions': pre_predictions}
    estimator_spec = score_sigmoid_n_train(pre_predictions, mode, labels=labels, params=params, to_predict=to_predict)
    return estimator_spec