from __future__ import print_function
import tensorflow as tf
from densenet import densenet161, densenet_arg_scope
import tensorflow.contrib.slim as slim
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

EPSILON = 1e-10
NUMS_CLASS = 20

def MLP(name, inputs, nums_in, nums_out):
    inputs = tf.layers.flatten(inputs)
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        W = tf.get_variable("W", [nums_in, nums_out], initializer=tf.random_normal_initializer(stddev=0.08))
        b = tf.get_variable("b", [nums_out], initializer=tf.constant_initializer(0.))
        inputs = tf.matmul(inputs, W) + b
    return inputs

def predict(img, k=3):
    imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
    labels = tf.placeholder(tf.int32, [None])
    lr = tf.placeholder(tf.float32)
    arg_scope = densenet_arg_scope()
    with slim.arg_scope(arg_scope):
        net, _ = densenet161((imgs-np.reshape(np.array([123.68, 116.779, 103.939]), [1, 1, 1, 3]))*0.017, num_classes=1000, is_training=False, reuse=tf.AUTO_REUSE)
    logits = MLP("logits", net, 2208, NUMS_CLASS)
    prediction = tf.nn.softmax(logits)
    values, indices = tf.nn.top_k(prediction, k)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="densenet161"))
    saver.restore(sess, "tf-densenet161.ckpt")
    saver = tf.train.Saver()
    saver.restore(sess, "./save_para/model.ckpt")
    metadata = sio.loadmat("./metadata.mat")
    img = img[np.newaxis, :, :, :]
    [VALUES, INDICES] = sess.run([values, indices], feed_dict={imgs: img})
    for i in range(k):
        print(metadata[str(INDICES[0, i])], end=":")
        print("%f"%(VALUES[0, i]))
    plt.imshow(img[0, :, :, :])
    plt.show()


if __name__ == "__main__":
    img = np.array(Image.open("/Users/wangzizhe/Desktop/final_pro/code/data/14.jpg").resize([224, 224]))
    predict(img, 3)


