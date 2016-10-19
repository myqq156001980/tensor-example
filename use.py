import tensorflow as tf
from PIL import Image
import numpy as np


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(convx, convw):
    return tf.nn.conv2d(convx, convw, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(poolx):
    return tf.nn.max_pool(poolx, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def classify(finename):
    sess = tf.Session()
    x = tf.placeholder("float", shape=[784])

    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])

    x_image = tf.reshape(x, [28, 28, 1])

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    keep_prob = tf.placeholder("float")
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    saver = tf.train.Saver()
    saver.restore(sess, './convolution_model.ckpt')

    im = Image.open(finename)
    pic_size = im.size
    source = im.split()
    grey_path = source[0]
    grey_px = grey_path.load()
    grey_list = [grey_px[j, i] for i in range(pic_size[0]) for j in range(pic_size[1])]
    na = np.array(grey_list)
    ix = np.multiply(na, 1.0 / 255.0)

    result = tf.argmax(y_conv, 1)
    index = sess.run(result, feed_dict={x: ix, keep_prob: 1.0})
    print ("The image is number: {num} ".format(num=index))
