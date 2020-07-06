# ==============================================================================
# Copyright 2019 The Project Author. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Implementation of Utils."""

import re
import numpy as np
import tensorflow as tf

def filter_variables(variables, variable_regex, is_whitelist):
    """Filter a list of variables based on the regex.
    Args:
        variables: a list of tf.Variable to be filtered.
        variable_regex: a regex specifying the filtering rule.
        is_whitelist: a bool. If True, indicate `variable_regex` specifies the
            variables to keep. If False, indicate `variable_regex` specfieis the
            variables to discard.
    Returns:
        filtered_variables: a list of tf.Variable after filtering.
    """
    if is_whitelist:
        filtered_variables = [
            v for v in variables if variable_regex is None or
            re.match(variable_regex, v.name)
        ]
    else:
        filtered_variables = [
            v for v in variables if variable_regex is None or
            not re.match(variable_regex, v.name)
        ]
    return filtered_variables

def filter_trainable_variables(variables, frozen_variable_prefix):
    """Filter and retrun trainable variables."""
    return filter_variables(
        variables, frozen_variable_prefix, is_whitelist=False)

def filter_regularization_variables(variables, regularization_variable_regex=r'.*(kernel|weight):0$'):
    """Filter and return regularization variables."""
    return filter_variables(
        variables, regularization_variable_regex, is_whitelist=True)

def regularization_loss(loss_factor):
    variables = tf.trainable_variables()
    var_list = filter_regularization_variables(variables)
    return loss_factor * tf.add_n([tf.nn.l2_loss(v) for v in var_list])

def global_avg_pool2d(inputs, axis=None):
    if axis is None:
        axis = [1, 2]
    outputs = tf.reduce_mean(inputs, axis=axis, keepdims=True)
    return outputs

def global_max_pool2d(inputs, axis=None):
    if axis is None:
        axis = [1, 2]
    outputs = tf.reduce_max(inputs, axis=axis, keepdims=True)
    return outputs

def fixed_padding(inputs, kernel_size):
    """Pads the input along the spatial dimensions independently of input size.
    Args:
        inputs: `Tensor` of size `[batch, channels, height, width]` or `[batch,
            height, width, channels]` depending on `data_format`.
        kernel_size: `int` kernel size to be used for `conv2d` or max_pool2d`
        operations. Should be a positive integer.
    Returns:
        A padded `Tensor` of the same `data_format` with size either intact
        (if `kernel_size == 1`) or padded (if `kernel_size > 1`).
    """
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
    return padded_inputs

def conv2d(inputs, filters, kernel_size, strides, use_bias=False, bias_initial=0,
           explicit_padding=False, name=None):
    """Strided 2-D convolution with explicit padding.
    The padding is consistent and is based only on `kernel_size`, not on the
    dimensions of `inputs` (as opposed to using `tf.layers.conv2d` alone).
    Args:
        inputs: `Tensor` of size `[batch, channels, height_in, width_in]`.
        filters: `int` number of filters in the convolution.
        kernel_size: `int` size of the kernel to be used in the convolution.
        strides: `int` strides of the convolution.
        use_bias: 'bool' use bias or not.
        name: 'str' the op name to be assigned.
    Returns:
        A `Tensor` of shape `[batch, filters, height_out, width_out]`.
    """
    if explicit_padding:
        if strides > 1:
            inputs = fixed_padding(inputs, kernel_size)
        padding = ('SAME' if strides == 1 else 'VALID')
    else:
        padding = 'SAME'

    return tf.layers.conv2d(
        inputs=inputs,
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        use_bias=use_bias,
        kernel_initializer=tf.variance_scaling_initializer(),
        bias_initializer=tf.constant_initializer(bias_initial),
        name=name)

def depthwise_conv2d(inputs, kernel_size, strides, use_bias=False,
                     explicit_padding=False, name=None):
    """Strided 2-D depthwise convolution with explicit padding.
    The padding is consistent and is based only on `kernel_size`, not on the
    dimensions of `inputs` (as opposed to using `tf.layers.conv2d` alone).
    Args:
        inputs: `Tensor` of size `[batch, channels, height_in, width_in]`.
        kernel_size: `int` kernel size of the convolution.
        strides: `int` strides of the convolution.
        use_bias: 'bool' use bias or not.
        name: 'str' the op name to be assigned.
    Returns:
        A `Tensor` of shape `[batch, filters, height_out, width_out]`.
    """
    if explicit_padding:
        if strides > 1:
            inputs = fixed_padding(inputs, kernel_size)
        padding = ('SAME' if strides == 1 else 'VALID')
    else:
        padding = 'SAME'

    depthwise_conv = tf.keras.layers.DepthwiseConv2D(
        [kernel_size, kernel_size],
        strides=strides,
        padding=padding,
        use_bias=use_bias,
        name=name)
    return depthwise_conv(inputs)

def hard_sigmoid(x):
    return tf.nn.relu6(x + 3) * 0.16667

def hard_swish(x):
    with tf.name_scope('hard_swish'):
        return x * tf.nn.relu6(x + np.float32(3)) * np.float32(1.0 / 6.0)
