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
"""Implementation of Cross-Iteration BatchNormalization."""

import tensorflow as tf

class BatchNormalization(object):
    """Implementation of Cross-Iteration BatchNormalization"""
    def __init__(self, momentum=0.9, epsilon=1e-5, use_cbn=True, num_iteration=3,
                 num_ignored=36000, scope=None):
        self.momentum = momentum
        self.epsilon = epsilon
        if scope is None:
            scope = 'batch_normalization'
        self.scope = self._find_valid_scope(scope)
        self.use_cbn = use_cbn
        self.num_iteration = num_iteration
        self.num_ignore = num_ignored
        self.beta = None
        self.gama = None
        self.moving_mean = None
        self.moving_variance = None
        if use_cbn:
            self.iteration_count = None
            self.pre_mean = None
            self.pre_mean_squared = None
            self.pre_mean_grad = None
            self.pre_mean_squared_grad = None
            self.pre_weight = None

    def _find_valid_scope(self, scope):
        global_scope = tf.get_variable_scope().name
        valid_scope = scope
        index = 0
        while 'status' != 'done':
            try:
                if index > 0:
                    name = '{}_{}/beta:0'.format(scope, index)
                else:
                    name = '{}/beta:0'.format(scope)
                if global_scope:
                    name = '{}/{}'.format(global_scope, name)
                tf.get_default_graph().get_tensor_by_name(name)
                index += 1
            except KeyError:
                if index > 0:
                    valid_scope = '{}_{}'.format(scope, index)
                break
        return valid_scope

    def _build(self, inputs, weight):
        in_channels = inputs.get_shape().as_list()[3]
        with tf.variable_scope(self.scope):
            self.beta = tf.get_variable(
                'beta', shape=(in_channels,), trainable=True, initializer=tf.zeros_initializer())
            self.gama = tf.get_variable(
                'gama', shape=(in_channels,), trainable=True, initializer=tf.ones_initializer())
            self.moving_mean = tf.get_variable(
                'moving_mean', shape=(in_channels,), trainable=False, initializer=tf.zeros_initializer())
            self.moving_variance = tf.get_variable(
                'moving_variance', shape=(in_channels,), trainable=False, initializer=tf.ones_initializer())
            if self.use_cbn:
                self.iteration_count = tf.get_variable(
                    'iteration_count', shape=(1,), trainable=False, initializer=tf.zeros_initializer())
                self.pre_mean = tf.get_variable(
                    'pre_mean', shape=(self.num_iteration, in_channels),
                    trainable=False, initializer=tf.zeros_initializer())
                self.pre_mean_squared = tf.get_variable(
                    'pre_mean_squared', shape=(self.num_iteration, in_channels),
                    trainable=False, initializer=tf.zeros_initializer())
                self.pre_mean_grad = tf.get_variable(
                    'pre_mean_grad', shape=([self.num_iteration, ] + weight.get_shape().as_list()),
                    trainable=False, initializer=tf.zeros_initializer())
                self.pre_mean_squared_grad = tf.get_variable(
                    'pre_mean_squared_grad', shape=([self.num_iteration, ] + weight.get_shape().as_list()),
                    trainable=False, initializer=tf.zeros_initializer())
                self.pre_weight = tf.get_variable(
                    'pre_weight', shape=([self.num_iteration, ] + weight.get_shape().as_list()),
                    trainable=False, initializer=tf.zeros_initializer())

    def _update_moving_average(self, variable, value, momentum):
        return tf.assign(
            variable, momentum * variable + (1.0 - momentum) * value)

    def _batch_moments(self, inputs, axis):
        mean = tf.reduce_mean(inputs, axis=axis, keepdims=True)
        squared_difference = tf.squared_difference(inputs, mean)
        variance = tf.reduce_mean(squared_difference, axis=axis, keepdims=True)
        mean = tf.squeeze(mean)
        variance = tf.squeeze(variance)
        return mean, variance

    def _batch_normalization(self, inputs, mean, variance, beta, gama, epsilon):
        # std=(variance+epsilon)**0.5
        # Y=gama*((X-mean)/std)+beta
        # Y=gama*rstd*(X-mean)+beta
        rstd = tf.rsqrt(variance + epsilon)
        inv = gama * rstd
        return inv * (inputs - mean) + beta

    def _moments(self, inputs):
        mean = tf.reduce_mean(inputs, axis=[0, 1, 2], keepdims=True)
        squared_difference = tf.squared_difference(inputs, mean)
        variance = tf.reduce_mean(squared_difference, axis=[0, 1, 2], keepdims=True)
        mean_squared = tf.reduce_mean(tf.pow(inputs, 2.0), axis=[0, 1, 2], keepdims=True)
        return mean, variance, mean_squared

    def _normalization(self, inputs, mean, sigma_squared, beta, gama, epsilon):
        # std=(variance+epsilon)**0.5
        # Y=gama*((X-mean)/std)+beta
        # Y=gama*rstd*(X-mean)+beta
        rstd = tf.rsqrt(sigma_squared + epsilon)
        inv = gama * rstd
        return inv * (inputs - mean) + beta

    def cbn_update(self, inputs, weight):
        mean, variance, mean_squared = self._moments(inputs)
        mean_grad = tf.gradients(mean, weight)[0]
        mean_squared_grad = tf.gradients(mean_squared, weight)[0]
        mean = tf.squeeze(mean, axis=[1, 2])
        variance = tf.squeeze(variance, axis=[0, 1, 2])
        mean_squared = tf.squeeze(mean_squared, axis=[1, 2])

        mean_all = []
        mean_squared_all = []
        for k in range(self.num_iteration):
            mean_t = self.pre_mean[k, :] + tf.reduce_sum(
                self.pre_mean_grad[k, ...] * (weight - self.pre_weight[k, ...]), axis=[0, 1, 2])
            mean_all.append(tf.expand_dims(mean_t, axis=0))
            mean_squared_t = self.pre_mean_squared[k, :] + tf.reduce_sum(
                self.pre_mean_squared_grad[k, ...] * (weight - self.pre_weight[k, ...]), axis=[0, 1, 2])
            mean_squared_all.append(tf.expand_dims(mean_squared_t, axis=0))
        mean_all = [mean, ] + mean_all
        mean_squared_all = [mean_squared, ] + mean_squared_all
        mean_all = tf.concat(mean_all, axis=0)
        mean_squared_all = tf.concat(mean_squared_all, axis=0)
        sigma_squared_all = mean_squared_all - tf.pow(mean_all, 2.0)

        mean_all = tf.where(
            tf.less(sigma_squared_all, 0.0), tf.zeros_like(mean_all), mean_all)
        mean_squared_all = tf.where(
            tf.less(sigma_squared_all, 0.0), tf.zeros_like(mean_squared_all), mean_squared_all)
        mask = tf.cast(tf.greater_equal(sigma_squared_all, 0.0), dtype=tf.float32)
        count = tf.reduce_sum(mask, axis=0)
        count = tf.where(tf.less_equal(count, 0.0), tf.ones_like(count), count)
        mean_compensated = tf.reduce_mean(mean_all, axis=0) / count
        sigma_squared = tf.reduce_mean(mean_squared_all, axis=0) / count - tf.pow(mean_compensated, 2.0)

        pre_mean = self.pre_mean[:-1, :]
        pre_mean_update_op = tf.assign(
            self.pre_mean, tf.concat([mean, pre_mean], axis=0))
        mean = tf.squeeze(mean, axis=0)
        pre_mean_squared = self.pre_mean_squared[:-1, :]
        pre_mean_squared_update_op = tf.assign(
            self.pre_mean_squared, tf.concat([mean_squared, pre_mean_squared], axis=0))
        pre_mean_grad = self.pre_mean_grad[:-1, ...]
        mean_grad = tf.expand_dims(mean_grad, axis=0)
        pre_mean_grad_update_op = tf.assign(
            self.pre_mean_grad, tf.concat([mean_grad, pre_mean_grad], axis=0))
        pre_mean_squared_grad = self.pre_mean_squared_grad[:-1, ...]
        mean_squared_grad = tf.expand_dims(mean_squared_grad, axis=0)
        pre_mean_squared_grad_update_op = tf.assign(
            self.pre_mean_squared_grad, tf.concat([mean_squared_grad, pre_mean_squared_grad], axis=0))
        pre_weight = self.pre_weight[:-1, ...]
        weight = tf.expand_dims(weight, axis=0)
        pre_weight_update_op = tf.assign(
            self.pre_weight, tf.concat([weight, pre_weight], axis=0))
        iteration_count_update_op = tf.assign_add(self.iteration_count, [1.0])

        tf.add_to_collection(
            tf.GraphKeys.UPDATE_OPS, pre_mean_update_op)
        tf.add_to_collection(
            tf.GraphKeys.UPDATE_OPS, pre_mean_squared_update_op)
        tf.add_to_collection(
            tf.GraphKeys.UPDATE_OPS, pre_mean_grad_update_op)
        tf.add_to_collection(
            tf.GraphKeys.UPDATE_OPS, pre_mean_squared_grad_update_op)
        tf.add_to_collection(
            tf.GraphKeys.UPDATE_OPS, pre_weight_update_op)
        tf.add_to_collection(
            tf.GraphKeys.UPDATE_OPS, iteration_count_update_op)

        moving_mean_update_op = self._update_moving_average(
            self.moving_mean, mean_compensated, self.momentum)
        moving_variance_update_op = self._update_moving_average(
            self.moving_variance, sigma_squared, self.momentum)

        tf.add_to_collection(
            tf.GraphKeys.UPDATE_OPS, moving_mean_update_op)
        tf.add_to_collection(
            tf.GraphKeys.UPDATE_OPS, moving_variance_update_op)
        return mean, variance, mean_compensated, sigma_squared

    def bn_update(self, inputs):
        mean, variance = self._batch_moments(inputs, axis=[0, 1, 2])
        moving_mean_update_op = self._update_moving_average(
            self.moving_mean, mean, self.momentum)
        moving_variance_update_op = self._update_moving_average(
            self.moving_variance, variance, self.momentum)

        tf.add_to_collection(
            tf.GraphKeys.UPDATE_OPS, moving_mean_update_op)
        tf.add_to_collection(
            tf.GraphKeys.UPDATE_OPS, moving_variance_update_op)
        return mean, variance

    def __call__(self, inputs, weight, is_training=False):
        if self.beta is None:
            self._build(inputs, weight)

        if is_training and self.use_cbn:
            mean, variance, mean_compensated, sigma_squared = self.cbn_update(inputs, weight)
            iteration_count = self.iteration_count * tf.ones_like(mean)
            tag_count = self.num_ignore * tf.ones_like(mean)
            mean_compensated = tf.where(
                tf.less(iteration_count, tag_count), mean, mean_compensated)
            sigma_squared = tf.where(
                tf.less(iteration_count, tag_count), variance, sigma_squared)
            outputs = self._normalization(
                inputs, mean_compensated, sigma_squared, self.beta, self.gama, self.epsilon)
        else:
            if is_training:
                mean, variance = self.bn_update(inputs)
            else:
                mean, variance = self.moving_mean, self.moving_variance
            outputs = self._batch_normalization(
                inputs, mean, variance, self.beta, self.gama, self.epsilon)
        return outputs
