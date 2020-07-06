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
"""Implementation of NetWork."""

import tensorflow as tf
import utils as utils
from cbn import BatchNormalization

def network(inputs, use_bn=True, use_cbn=True, is_training=False):
    use_bias = not use_bn
    net = utils.conv2d(inputs, 16, kernel_size=3, strides=1, use_bias=use_bias)
    if use_bn:
        weight = tf.get_default_graph().get_tensor_by_name('conv2d/kernel:0')
        net = BatchNormalization(use_cbn=use_cbn)(net, weight, is_training)
    net = tf.nn.relu(net)
    net = utils.conv2d(net, 24, kernel_size=3, strides=2, use_bias=use_bias)
    if use_bn:
        weight_1 = tf.get_default_graph().get_tensor_by_name('conv2d_1/kernel:0')
        net = BatchNormalization(use_cbn=use_cbn)(net, weight_1, is_training)
    net = tf.nn.relu(net)
    net = utils.conv2d(net, 24, kernel_size=3, strides=1, use_bias=use_bias)
    if use_bn:
        weight_2 = tf.get_default_graph().get_tensor_by_name('conv2d_2/kernel:0')
        net = BatchNormalization(use_cbn=use_cbn)(net, weight_2, is_training)
    net = tf.nn.relu(net)
    net = utils.conv2d(net, 32, kernel_size=3, strides=2, use_bias=use_bias)
    if use_bn:
        weight_3 = tf.get_default_graph().get_tensor_by_name('conv2d_3/kernel:0')
        net = BatchNormalization(use_cbn=use_cbn)(net, weight_3, is_training)
    net = tf.nn.relu(net)
    net = utils.conv2d(net, 48, kernel_size=3, strides=1, use_bias=use_bias)
    if use_bn:
        weight_4 = tf.get_default_graph().get_tensor_by_name('conv2d_4/kernel:0')
        net = BatchNormalization(use_cbn=use_cbn)(net, weight_4, is_training)
    net = tf.nn.relu(net)
    net = utils.global_avg_pool2d(net)
    net = utils.conv2d(net, 10, kernel_size=3, strides=1, use_bias=True)
    logits = tf.squeeze(net, axis=[1, 2])
    return logits
