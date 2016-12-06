import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
from tensorflow.python.ops import variable_scope as vs
import numpy as np
from rio import load_id2vec
import collections


class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(
      self, B, sequence_length, sequence_length1, sequence_length2, sequence_length3, sequence_length4,
      num_classes, vocab_size,
      embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):

        ######## cnn ########

        # Placeholders for input, output and dropout
        self.B = B
        self.input_x = tf.placeholder(tf.int32, [B, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [B, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # 3 seg lstm
        self.input_seg1 = tf.placeholder(tf.int32, [B, sequence_length1], name="input_seg1")
        self.input_seg2 = tf.placeholder(tf.int32, [B, sequence_length2], name="input_seg2")
        self.input_seg3 = tf.placeholder(tf.int32, [B, sequence_length3], name="input_seg3")
        self.input_seg4 = tf.placeholder(tf.int32, [B, sequence_length4], name="input_seg4")

        self.entities = tf.placeholder(tf.int32, [B, 2], name="entities")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.name_scope("embedding"):
            # temp = tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0)
            # W = tf.Variable(
            #     tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
            #     name="W")

            id2vec = load_id2vec("../data/id2vec.bin")

            # add marker
            # replace two entities with two markers

            od = [item[1] for item in sorted(id2vec.items())]
            od.insert(0,[0]*50)
            id2vec1 = np.array(od, dtype=np.float32)
            #id2vec1.astype(np.float32)

            # init = tf.constant(id2vec1)
            # temp1 = tf.get_variable('embedding',initializer = init)
            # tf.Variable(tf.convert_to_tensor(np.eye(784), dtype=tf.float32))
            # temp1 = tf.Variable(initial_value=id2vec1)
            _W = tf.Variable(
                tf.convert_to_tensor(id2vec1, dtype=tf.float32),
                name="_W")
            #marker1 = tf.Variable(np.zeros([1, id2vec1.shape[1]], dtype=np.float32), trainable=False)
            #marker2 = tf.Variable(np.zeros([1, id2vec1.shape[1]], dtype=np.float32), trainable=False)
            #padding = tf.Variable(np.zeros([1, id2vec1.shape[1]], dtype=np.float32), trainable=False)
            marker1 = tf.Variable(np.zeros([1, id2vec1.shape[1]], dtype=np.float32), trainable=True)
            marker2 = tf.Variable(np.zeros([1, id2vec1.shape[1]], dtype=np.float32), trainable=True)
            padding = tf.Variable(np.zeros([1, id2vec1.shape[1]], dtype=np.float32), trainable=True)
            W = tf.concat(0, [_W, marker1, marker2, padding], name='W')

            self.embedded_chars = tf.nn.embedding_lookup(W, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

            self.x1 = tf.nn.embedding_lookup(W, self.input_seg1)
            self.x2 = tf.nn.embedding_lookup(W, self.input_seg2)
            self.x3 = tf.nn.embedding_lookup(W, self.input_seg3)
            self.x4 = tf.nn.embedding_lookup(W, self.input_seg4)

            self.e = tf.nn.embedding_lookup(W, self.entities)
            self.e = tf.reshape(self.e, [B, -1])

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        c_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                # (B, n-k+1, 1, m)
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")

                """ Maxpooling over the outputs """
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")

                pooled_outputs.append(pooled)


                """ attention
                # (B, n-k+1, m)
                h = tf.squeeze(h)
                # (B*(n-k+1), m)
                _, n_k_1, _ = h.get_shape()
                n_k_1 = n_k_1.value
                h = tf.reshape(h, [self.B*n_k_1, num_filters])

                # (m, m)
                w1 = tf.Variable(tf.truncated_normal([num_filters, num_filters], stddev=0.1), name="w1")
                b1 = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b1")
                # (B*(n-k+1), m)
                hw1 = tf.squeeze(tf.matmul(h, w1))
                hw1 = tf.nn.bias_add(hw1, b1, name='hw1_b1')
                # (B*(n-k+1), m)
                tanh_hw1 = tf.tanh(hw1, 'tanh_hw1')

                # (m, 1)
                v1 = tf.Variable(tf.truncated_normal([num_filters, 1], stddev=0.1), name='v1')
                # (B*(n-k+1), 1)
                e = tf.matmul(tanh_hw1, v1, name='e')
                # (B, n-k+1)
                e = tf.reshape(e, [self.B, n_k_1], name='e')

                # (B, n-k+1)
                alpha = tf.nn.softmax(e, name="alpha")
                # (B, n-k+1, 1)
                alpha = tf.reshape(alpha, [self.B, n_k_1, 1], name="alpha")

                # (B, n-k+1, m)
                h = tf.reshape(h, [self.B, n_k_1, num_filters])
                # (B, m, n-k+1)
                ht = tf.transpose(h, [0, 2, 1], name="v")

                # attention
                # (B, m, 1)
                c = tf.batch_matmul(ht, alpha, name="c")
                # (B, m)
                c = tf.squeeze(c, name="c")

                c_outputs.append(c)
                """

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(3, pooled_outputs)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # attention
        #self.c_outputs = tf.concat(1, c_outputs)
        #self.c_outputs_flat = tf.reshape(self.c_outputs, [-1, num_filters_total])

        ######## lstm ########
        n_lstm_hidden = 50
        def mylstm(scope, input_x, n_hidden=n_lstm_hidden):
            # returns output, state
            _, n_steps, n_input = input_x.get_shape()
            n_steps, n_input = n_steps.value, n_input.value
            x = tf.transpose(input_x, [1, 0, 2])
            x = tf.reshape(x, [-1, n_input])
            x = tf.split(0, n_steps, x)
            lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)
            output, _ = rnn.rnn(lstm_cell, x, dtype=tf.float32, scope=scope)
            return output[-1]

        lstm_outputs = []
        lstm_outputs.append(mylstm('lstm1', self.x1))
        lstm_outputs.append(mylstm('lstm4', self.x4))
        lstm_outputs.append(mylstm('lstm2', self.x2))
        lstm_outputs.append(mylstm('lstm3', self.x3))

        # concat
        self.lstm_outputs = tf.concat(1, lstm_outputs)

        # Final forward
        e_shape = self.e.get_shape()[1].value
        # lstm + cnn
        self.final_outputs = tf.concat(1, [self.lstm_outputs, self.h_pool_flat, self.e])
        self.final_outputs = tf.concat(1, [self.lstm_outputs, self.h_pool_flat])
        final_weight_shape = (n_lstm_hidden * 4 + num_filters_total, num_classes)

        # cnn only
        # self.final_outputs = tf.concat(1, [self.h_pool_flat])
        # final_weight_shape = (num_filters_total, num_classes)

        final_bias_shape = (num_classes,)

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.final_outputs, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=final_weight_shape,
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=final_bias_shape), name="b")

            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")
            self.groundtruth = tf.argmax(self.input_y, 1)

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(self.scores, self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
