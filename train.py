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
"""Implementation of Train."""

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
from utils import regularization_loss
from network import network
from cifar10 import get_CIFAR10_data

def losses(labels, logits, l2_factor=0.00001):
    cls_loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    cls_loss = tf.reduce_mean(cls_loss)
    l2_loss = regularization_loss(l2_factor)
    return cls_loss, l2_loss

def shuffle(x, y):
    index = np.random.permutation(y.shape[0])
    x = x[index, ...]
    y = y[index, :]
    return x, y

def main():
    mnist_path = 'MNIST'
    cifar10_path = 'CIFAR10'
    model_path = 'model'
    momentum = 0.9
    learning_rate_init = 0.01
    batch_size = 4
    train_epochs = 10
    mnist = None
    use_cifar10 = True
    X_train = None
    X_test = None
    y_train = None
    y_test = None

    if use_cifar10:
        cifar10 = get_CIFAR10_data(cifar10_path)
        X_train = cifar10['X_train']
        y_train = np.eye(10)[cifar10['y_train']]
        index = np.random.permutation(y_train.shape[0])
        X_train = X_train[index, ...]
        y_train = y_train[index, :]
        X_test = cifar10['X_val']
        y_test = np.eye(10)[cifar10['y_val']]

    else:
        mnist = input_data.read_data_sets(mnist_path, one_hot=True)

    if use_cifar10:
        inputs = tf.placeholder(tf.float32, (None, None, None, 3), name='inputs')
        labels = tf.placeholder(tf.float32, (None, 10), name='labels')
    else:
        inputs = tf.placeholder(tf.float32, (None, 784), name='inputs')
        labels = tf.placeholder(tf.float32, (None, 10), name='labels')
        inputs = tf.reshape(inputs, shape=(-1, 28, 28, 1))
    logits = network(inputs, use_bn=True, use_cbn=True, is_training=True)
    cls_loss, l2_loss = losses(labels, logits)
    total_loss = cls_loss + l2_loss

    outputs = tf.nn.softmax(logits)
    accuracy_condition = tf.equal(tf.argmax(outputs, axis=-1), tf.argmax(labels, axis=-1))
    accuracy_op = tf.reduce_mean(tf.cast(accuracy_condition, tf.float32))

    global_step = tf.Variable(1.0, dtype=tf.float32, trainable=False, name='global_step')
    if use_cifar10:
        num_batchs = y_train.shape[0] // batch_size
    else:
        num_batchs = mnist.train.num_examples // batch_size
    total_steps = train_epochs * num_batchs
    learning_rate = learning_rate_init * tf.cos(global_step / total_steps * np.pi / 2.0)
    update_global_step_op = tf.assign_add(global_step, 1.0)

    weight = tf.get_default_graph().get_tensor_by_name('conv2d/kernel:0')
    gradient = tf.gradients(logits, weight)[0]
    gradient_norm = learning_rate * tf.norm(gradient, ord='euclidean')
    gradient_op = gradient_norm / tf.norm(weight, ord='euclidean')

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)
    optimizer_op = optimizer.minimize(total_loss)
    train_ops = tf.group([optimizer_op, update_ops])

    for var in tf.global_variables():
        print('=>' + var.op.name)
    print('start network training...')
    test_accuracy = []
    train_loss = []
    saver = tf.train.Saver(max_to_keep=5)
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(1, train_epochs + 1):
            for batch in range(num_batchs):
                if use_cifar10:
                    batch_inputs = X_train[batch:(batch+batch_size), ...]
                    batch_labels = y_train[batch:(batch+batch_size), :]
                else:
                    batch_inputs, batch_labels = mnist.train.next_batch(batch_size)
                _ = sess.run(
                    train_ops, feed_dict={inputs: batch_inputs, labels: batch_labels})
                sess.run(update_global_step_op)
                if batch % 100 == 0:
                    gradient_np = sess.run(
                        gradient_op, feed_dict={inputs: batch_inputs, labels: batch_labels})
                    print('batch: %d, norm of the gradient: %.5f' % (batch, gradient_np))
            if use_cifar10:
                batch_inputs = X_test
                batch_labels = y_test
            else:
                batch_inputs = mnist.test.images
                batch_labels = mnist.test.labels
            loss, accuracy = sess.run(
                [total_loss, accuracy_op],
                feed_dict={inputs: batch_inputs, labels: batch_labels})
            train_loss.append(loss)
            test_accuracy.append(accuracy)
            print('epoch: %d, loss: %.5f, accuracy: %.5f' % (epoch, loss, accuracy))
        saver.save(sess, '%s/model-%.5f.ckpt' % (model_path, accuracy), global_step=epoch)

    steps = range(train_epochs)
    plt.subplot(211)
    plt.plot(steps, train_loss, 'k-')
    plt.title('softmax loss over epochs')
    plt.xlabel('epoch')
    plt.ylabel('softmax loss')
    plt.subplot(212)
    plt.plot(steps, test_accuracy, 'b-')
    plt.title('test accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.show()

if __name__ == '__main__':
    main()
