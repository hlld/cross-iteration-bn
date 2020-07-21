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
"""Implementation of Evaluation."""

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from network import network
from cifar10 import get_CIFAR10_data

def main():
    mnist_path = 'MNIST'
    cifar10_path = 'CIFAR10'
    model_path = 'model'
    mnist = None
    use_cifar10 = True
    X_test = None
    y_test = None

    if use_cifar10:
        cifar10 = get_CIFAR10_data(cifar10_path)
        X_test = cifar10['X_test']
        y_test = np.eye(10)[cifar10['y_test']]
    else:
        mnist = input_data.read_data_sets(mnist_path, one_hot=True)

    if use_cifar10:
        inputs = tf.placeholder(tf.float32, (None, None, None, 3), name='inputs')
        labels = tf.placeholder(tf.float32, (None, 10), name='labels')
    else:
        inputs = tf.placeholder(tf.float32, (None, 784), name='inputs')
        labels = tf.placeholder(tf.float32, (None, 10), name='labels')
        inputs = tf.reshape(inputs, shape=(-1, 28, 28, 1))
    logits = network(inputs, use_bn=True, use_cbn=True, is_training=False)

    outputs = tf.nn.softmax(logits)
    accuracy_condition = tf.equal(tf.argmax(outputs, axis=-1), tf.argmax(labels, axis=-1))
    accuracy_op = tf.reduce_mean(tf.cast(accuracy_condition, tf.float32))

    for var in tf.global_variables():
        print('=> variable ' + var.op.name)
    print('start network evaluation...')
    loader = tf.train.Saver()
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        try:
            print('=> restoring weights from: %s ...' % model_path)
            ckpt = tf.train.latest_checkpoint(model_path)
            loader.restore(sess, ckpt)
        except:
            print('=> restoring weights from: %s failed.' % model_path)
        if use_cifar10:
            batch_inputs = X_test
            batch_labels = y_test
        else:
            batch_inputs = mnist.test.images
            batch_labels = mnist.test.labels
        accuracy = sess.run(accuracy_op, feed_dict={
            inputs: batch_inputs, labels: batch_labels})
        print('=> accuracy: %.5f' % accuracy)

if __name__ == '__main__':
    main()
