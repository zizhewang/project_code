import tensorflow as tf
# from inception_v2 import inception_v2_arg_scope, inception_v2
from densenet import densenet161, densenet_arg_scope
import tensorflow.contrib.slim as slim
import scipy.io as sio
import numpy as np

NUMS_CLASS = 20
BATCH_SIZE = 64
EPOCHES = 90
EPSILON = 1e-10
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4


def MLP(name, inputs, nums_in, nums_out):
    inputs = tf.layers.flatten(inputs)
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        W = tf.get_variable("W", [nums_in, nums_out], initializer=tf.random_normal_initializer(stddev=0.08))
        b = tf.get_variable("b", [nums_out], initializer=tf.constant_initializer(0.))
        inputs = tf.matmul(inputs, W) + b
    return inputs


def preprocess(data):
    return (data / 255.0 - 0.5) * 2


def test(sess, accuracy, imgs, labels, TEST_DATA, LABELS):
    NUMS = LABELS.shape[0]
    c = 0
    acc = 0
    for i in range(NUMS // BATCH_SIZE - 1):
        acc += sess.run(accuracy, feed_dict={imgs: TEST_DATA[i * BATCH_SIZE:i * BATCH_SIZE + BATCH_SIZE],
                                             labels: LABELS[i * BATCH_SIZE:i * BATCH_SIZE + BATCH_SIZE]})
        c += 1
    return acc / c


def main():
    imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
    labels = tf.placeholder(tf.int32, [None])
    lr = tf.placeholder(tf.float32)
    arg_scope = densenet_arg_scope()
    with slim.arg_scope(arg_scope):
        net, _ = densenet161((imgs-np.reshape(np.array([123.68, 116.779, 103.939]), [1, 1, 1, 3]))*0.017, num_classes=1000, is_training=False, reuse=tf.AUTO_REUSE)
    logits = MLP("logits", net, 2208, NUMS_CLASS)
    prediction = tf.nn.softmax(logits)
    onehot_label = tf.one_hot(labels, NUMS_CLASS)
    accuracy = tf.reduce_mean(
        tf.cast(tf.equal(tf.arg_max(prediction, dimension=1), tf.arg_max(onehot_label, dimension=1)), dtype=tf.float32))
    loss = tf.reduce_mean(-tf.log(tf.reduce_sum(prediction * onehot_label, axis=1) + EPSILON))
    trainable_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="logits")
#     regular = tf.add_n([tf.nn.l2_loss(var) for var in trainable_variables])
    Opt = tf.train.AdamOptimizer(lr).minimize(loss, var_list=trainable_variables)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="densenet161"))
    saver.restore(sess, "tf-densenet161.ckpt")
    saver = tf.train.Saver()
    saver.restore(sess, "./save_para/model.ckpt")

    dataset = sio.loadmat("./train_test.mat")
    train_data, train_labels, test_data, test_labels = dataset["train_data"], dataset["train_labels"], dataset["test_data"], dataset["test_labels"]
    train_labels = np.squeeze(train_labels)
    test_labels = np.squeeze(test_labels)
    test_acc = test(sess, accuracy, imgs, labels, test_data, test_labels)
    LR = LEARNING_RATE
    TRAIN_NUMS = train_labels.shape[0]
    print("Test accuracy: %f" % (test_acc))
    for i in range(EPOCHES):
        if i == 0 or i == 30:
            LR /= 10
        for j in range(TRAIN_NUMS // BATCH_SIZE - 1):
            BATCH = train_data[j * BATCH_SIZE:j * BATCH_SIZE + BATCH_SIZE]
            BATCH_LABELS = train_labels[j * BATCH_SIZE:j * BATCH_SIZE + BATCH_SIZE]
            sess.run(Opt, feed_dict={imgs: BATCH, labels: BATCH_LABELS, lr: LR})
            if j % 10 == 0:
                [LOSS, ACC] = sess.run([loss, accuracy], feed_dict={imgs: BATCH, labels: BATCH_LABELS})
                print("Epoch: %d, Iteration: %d, Loss: %f, Train accuracy: %f" % (i, j, LOSS, ACC))
        saver.save(sess, "./save_para/model.ckpt")
        test_acc = test(sess, accuracy, imgs, labels, test_data, test_labels)
        print("Test accuracy: %f" % (test_acc))
        rand_nums = np.arange(0, TRAIN_NUMS)
        np.random.shuffle(rand_nums)
        train_data = train_data[rand_nums]
        train_labels = train_labels[rand_nums]


if __name__ == "__main__":
    main()


