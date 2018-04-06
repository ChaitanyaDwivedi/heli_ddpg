import tensorflow as tf
import numpy as np

class nn:
    def __init__(self):
        self.model = self._build_Model()

    def _build_Model(self):
        graph = tf.Graph()
        with graph.as_default():
            initializer = tf.truncated_normal
            #initializer = tf.contrib.layers.xavier_initializer()
            voltages_and_angles = tf.placeholder(tf.float32, [None, 5])
            correct_angles = tf.placeholder(tf.float32, [None, 3])
            w1 = tf.Variable(initializer([5, 10]))
            b1 = tf.Variable(tf.zeros([10]))
            h1 = tf.nn.tanh(tf.matmul(voltages_and_angles, w1) + b1)

            w2 = tf.Variable(initializer([10, 10]))
            b2 = tf.Variable(tf.zeros([10]))
            h2 = tf.nn.tanh(tf.matmul(h1, w2) + b2)

            w3 = tf.Variable(initializer([10, 10]))
            b3 = tf.Variable(tf.zeros([10]))
            h3 = tf.nn.tanh(tf.matmul(h2, w3) + b3)

            w4 = tf.Variable(initializer([10, 10]))
            b4 = tf.Variable(tf.zeros([10]))
            h4 = tf.nn.tanh(tf.matmul(h3, w4) + b4)

            w5 = tf.Variable(initializer([10, 10]))
            b5 = tf.Variable(tf.zeros([10]))
            h5 = tf.nn.tanh(tf.matmul(h4, w5) + b5)

            w_final = tf.Variable(initializer([10, 3]))
            b_final = tf.Variable(tf.zeros([3]))
            output = tf.matmul(h5, w_final) + b_final
            saver = tf.train.Saver()
        model = {}
        model['graph'] = graph
        model['inp_ph'] = voltages_and_angles
        model['out_ph'] = correct_angles
        model['predictions'] = output
        model['saver'] = saver
        return model

    def predict(self, v1, v2, current):
        inp = np.array([current[0], current[1], current[2], v1, v2]).reshape(1,5)
        sess = tf.Session(graph = self.model["graph"])
        self.model['saver'].restore(sess, "./DiveshSimpleNN_5_to_3_Dirty")
        prediction = sess.run(self.model['predictions'], feed_dict = {self.model['inp_ph']: inp})
        return prediction
