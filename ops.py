# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import
import tensorflow as tf

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import confusion_matrix
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import weights_broadcast_ops


def batch_norm(x, is_training, epsilon=1e-5, decay=0.9, scope="batch_norm",
               parameter_update_device='-1'):
    with tf.device(parameter_update_device):
        var = tf.contrib.layers.batch_norm(x, decay=decay, updates_collections=None, epsilon=epsilon,
                                           scale=True, is_training=is_training, scope=scope)

    return var

def layer_norm(x, scope="layer_norm",
               parameter_update_device='-1'):
    with tf.device(parameter_update_device):
        var = tf.contrib.layers.layer_norm(x,scope=scope)

    return var


def conv2d(x,
           output_filters,
           weight_decay_rate,
           kh=5, kw=5, sh=2, sw=2,
           initializer='None',
           scope="conv2d",
           parameter_update_device='-1',
           weight_decay=False,
           name_prefix='None',
           padding='SAME'):
    weight_stddev = 0.02
    bias_init_stddev = 0.0
    with tf.variable_scope(scope):
        shape = x.get_shape().as_list()

        if initializer == 'NormalInit':
            W = variable_creation_on_device(name='W',
                                            shape=[kh, kw, shape[-1], output_filters],
                                            initializer=tf.truncated_normal_initializer(stddev=weight_stddev),
                                            parameter_update_device=parameter_update_device)
        elif initializer == 'XavierInit':
            W = variable_creation_on_device(name='W',
                                            shape=[kh, kw, shape[-1], output_filters],
                                            initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                            parameter_update_device=parameter_update_device)

        if weight_decay:
            weight_decay = tf.multiply(tf.nn.l2_loss(W), weight_decay_rate, name='weight_decay')
            if not name_prefix.find('/') == -1:
                tf.add_to_collection(name_prefix[0:name_prefix.find('/')] + '_weight_decay', weight_decay)
            else:
                tf.add_to_collection(name_prefix + '_weight_decay', weight_decay)

        Wconv = tf.nn.conv2d(x, W, strides=[1, sh, sw, 1], padding=padding)

        biases = variable_creation_on_device('b',
                                             shape=[output_filters],
                                             initializer=tf.constant_initializer(bias_init_stddev),
                                             parameter_update_device=parameter_update_device)

        Wconv_plus_b = tf.reshape(tf.nn.bias_add(Wconv, biases), Wconv.get_shape())

        return Wconv_plus_b





def lrelu(x, leak=0.2):
    return tf.maximum(x, leak * x)

def relu (x):
    return tf.nn.relu(features=x)





def dropout(x,drop_v):
    output = tf.nn.dropout(x,keep_prob=1-drop_v)
    return output


def fc(x,
       output_size,weight_decay_rate,
       scope="fc",
       initializer='None',
       parameter_update_device='-1',
       name_prefix='None',
       weight_decay=False):
    weight_stddev = 0.02
    bias_stddev = 0.0
    with tf.variable_scope(scope):
        shape = x.get_shape().as_list()

        if initializer == 'NormalInit':
            W = variable_creation_on_device(name="W",
                                            shape=[shape[1], output_size],
                                            initializer=tf.random_normal_initializer(stddev=weight_stddev),
                                            parameter_update_device=parameter_update_device)
        elif initializer =='XavierInit':
            W = variable_creation_on_device(name="W",
                                            shape=[shape[1], output_size],
                                            initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                            parameter_update_device=parameter_update_device)

        if weight_decay:
            weight_decay = tf.multiply(tf.nn.l2_loss(W), weight_decay_rate, name='weight_decay')
            if not name_prefix.find('/') == -1:
                tf.add_to_collection(name_prefix[0:name_prefix.find('/')] + '_weight_decay', weight_decay)
            else:
                tf.add_to_collection(name_prefix + '_weight_decay', weight_decay)


        b = variable_creation_on_device("b", shape=[output_size],
                                        initializer=tf.constant_initializer(bias_stddev),
                                        parameter_update_device=parameter_update_device)
        return tf.matmul(x, W) + b


def variable_creation_on_device(name,
                                shape,
                                initializer,
                                parameter_update_device='-1'):
    """Helper to create a Variable stored on CPU memory.
    Args:
      name: name of the variable
      shape: list of ints
      initializer: initializer for Variable
    Returns:
      Variable Tensor
    """
    with tf.device(parameter_update_device):
        dtype = tf.float32
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
    return var


def _remove_squeezable_dimensions(predictions, labels, weights):
    predictions = ops.convert_to_tensor(predictions)
    if labels is not None:
        labels, predictions = confusion_matrix.remove_squeezable_dimensions(labels, predictions)
        predictions.get_shape().assert_is_compatible_with(labels.get_shape())

    if weights is None:
        return predictions, labels, None

    weights = ops.convert_to_tensor(weights)
    weights_shape = weights.get_shape()
    weights_rank = weights_shape.ndims
    if weights_rank == 0:
        return predictions, labels, weights

    predictions_shape = predictions.get_shape()
    predictions_rank = predictions_shape.ndims
    if (predictions_rank is not None) and (weights_rank is not None):
        # Use static rank.
        if weights_rank - predictions_rank == 1:
            weights = array_ops.squeeze(weights, [-1])
        elif predictions_rank - weights_rank == 1:
            weights = array_ops.expand_dims(weights, [-1])
    else:
        # Use dynamic rank.
        weights_rank_tensor = array_ops.rank(weights)
        rank_diff = weights_rank_tensor - array_ops.rank(predictions)

        def _maybe_expand_weights():
            return control_flow_ops.cond(math_ops.equal(rank_diff, -1),
                                         lambda: array_ops.expand_dims(weights, [-1]), lambda: weights)

        # Don't attempt squeeze if it will fail based on static check.
        if (weights_rank is not None) and (not weights_shape.dims[-1].is_compatible_with(1)):
            maybe_squeeze_weights = lambda: weights
        else:
            maybe_squeeze_weights = lambda: array_ops.squeeze(weights, [-1])

        def _maybe_adjust_weights():
            return control_flow_ops.cond(math_ops.equal(rank_diff, 1), maybe_squeeze_weights, _maybe_expand_weights)

        # If weights are scalar, do nothing. Otherwise, try to add or remove a
        # dimension to match predictions.
        weights = control_flow_ops.cond(math_ops.equal(weights_rank_tensor, 0), lambda: weights, _maybe_adjust_weights)
    return predictions, labels, weights


def _confusion_matrix_at_thresholds(labels, predictions, thresholds, weights=None):
    with ops.control_dependencies(
            [check_ops.assert_greater_equal(predictions, math_ops.cast(0.0, dtype=predictions.dtype),
                                            message='predictions must be in [0, 1]'),
             check_ops.assert_less_equal(predictions, math_ops.cast(1.0, dtype=predictions.dtype),
                                         message='predictions must be in [0, 1]')]):
        predictions, labels, weights = _remove_squeezable_dimensions(
            predictions=math_ops.to_float(predictions),
            labels=math_ops.cast(labels, dtype=dtypes.bool),
            weights=weights)

    num_thresholds = len(thresholds)

    # Reshape predictions and labels.
    predictions_2d = array_ops.reshape(predictions, [-1, 1])
    labels_2d = array_ops.reshape(math_ops.cast(labels, dtype=dtypes.bool), [1, -1])

    # Use static shape if known.
    num_predictions = predictions_2d.get_shape().as_list()[0]

    # Otherwise use dynamic shape.
    if num_predictions is None:
        num_predictions = array_ops.shape(predictions_2d)[0]
    thresh_tiled = array_ops.tile(
        array_ops.expand_dims(array_ops.constant(thresholds), [1]),
        array_ops.stack([1, num_predictions]))

    # Tile the predictions after threshold them across different thresholds.
    pred_is_pos = math_ops.greater(
        array_ops.tile(array_ops.transpose(predictions_2d), [num_thresholds, 1]), thresh_tiled)
    pred_is_neg = math_ops.logical_not(pred_is_pos)
    label_is_pos = array_ops.tile(labels_2d, [num_thresholds, 1])
    label_is_neg = math_ops.logical_not(label_is_pos)

    if weights is not None:
        weights = weights_broadcast_ops.broadcast_weights(
            math_ops.to_float(weights), predictions)
        weights_tiled = array_ops.tile(
            array_ops.reshape(weights, [1, -1]), [num_thresholds, 1])
        thresh_tiled.get_shape().assert_is_compatible_with(
            weights_tiled.get_shape())
    else:
        weights_tiled = None

    values = {}

    # tp
    is_true_positive = math_ops.to_float(math_ops.logical_and(label_is_pos, pred_is_pos))
    if weights_tiled is not None:
        is_true_positive *= weights_tiled
    values['tp'] = math_ops.reduce_sum(is_true_positive, 1)

    # fn
    is_false_negative = math_ops.to_float(math_ops.logical_and(label_is_pos, pred_is_neg))
    if weights_tiled is not None:
        is_false_negative *= weights_tiled
    values['fn'] = math_ops.reduce_sum(is_false_negative, 1)

    # tn
    is_true_negative = math_ops.to_float(math_ops.logical_and(label_is_neg, pred_is_neg))
    if weights_tiled is not None:
        is_true_negative *= weights_tiled
    values['tn'] = math_ops.reduce_sum(is_true_negative, 1)

    # fp
    is_false_positive = math_ops.to_float(math_ops.logical_and(label_is_neg, pred_is_pos))
    if weights_tiled is not None:
        is_false_positive *= weights_tiled
    values['fp'] = math_ops.reduce_sum(is_false_positive, 1)

    return values


def auc(labels, predictions, weights=None, num_thresholds=200, name=None, summation_method='trapezoidal'):
    

    with variable_scope.variable_scope(name, 'auc', (labels, predictions, weights)):

        kepsilon = 1e-7  # to account for floating point imprecisions
        thresholds = [(i + 1) * 1.0 / (num_thresholds - 1) for i in range(num_thresholds - 2)]
        thresholds = [0.0 - kepsilon] + thresholds + [1.0 + kepsilon]

        values = _confusion_matrix_at_thresholds(labels, predictions, thresholds, weights)

        # Add epsilons to avoid dividing by 0.
        epsilon = 1.0e-6

        def compute_auc(tp, fn, tn, fp, auc_name):
            """Computes the roc-auc or pr-auc based on confusion counts."""

            rec = math_ops.div(tp + epsilon, tp + fn + epsilon)
            fp_rate = math_ops.div(fp, fp + tn + epsilon)
            x = fp_rate
            y = rec

            if summation_method in ('trapezoidal', 'careful_interpolation'):
                return math_ops.reduce_sum(
                    math_ops.multiply(x[:num_thresholds - 1] - x[1:], (y[:num_thresholds - 1] + y[1:]) / 2.),
                    name=auc_name)
            elif summation_method == 'minoring':
                return math_ops.reduce_sum(
                    math_ops.multiply(
                        x[:num_thresholds - 1] - x[1:], math_ops.minimum(y[:num_thresholds - 1], y[1:])), name=auc_name)
            elif summation_method == 'majoring':
                return math_ops.reduce_sum(math_ops.multiply(
                    x[:num_thresholds - 1] - x[1:], math_ops.maximum(y[:num_thresholds - 1], y[1:])), name=auc_name)
            else:
                raise ValueError('Invalid summation_method: %s' % summation_method)

        auc_value = compute_auc(values['tp'], values['fn'], values['tn'], values['fp'], 'value')

        return auc_value