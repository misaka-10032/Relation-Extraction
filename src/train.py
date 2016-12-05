#! /usr/local/bin/python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn
from rio import load_data,load_id2vec

# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", .25, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("data_file", "../data/instances.bin", "Data source for the data.")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 50, "Dimensionality of character embedding (default: 50)")
#tf.flags.DEFINE_string("filter_sizes", "3", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_string("filter_sizes", "2,3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 50, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.001, "L2 regularizaion lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 100, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 100, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 60, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 60, "Save model after this many steps (default: 100)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")


# Data Preparatopn
# ==================================================

# Load data
print("Loading data...")
# x_text, y = data_helpers.load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file)
# print x_text
# print y
#
# # Build vocabulary
# max_document_length = max([len(x.split(" ")) for x in x_text])
# vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
# x = np.array(list(vocab_processor.fit_transform(x_text)))

sz = 8000
# sz = 200

vocabulary_size = 22132
marker1 = vocabulary_size
marker2 = vocabulary_size + 1
padding = vocabulary_size + 2

instances = load_data("../data/instances.bin")
x = [item[0] for item in instances[:sz]]
max_len = max([len(item) for item in x])
x = [list(item) + [padding]*(max_len - len(item)) for item in x]
x = np.array(x)

# cnn add marker
#for i, instance in enumerate(instances[:sz]):
#    p1, p2 = instance[1]
#    x[i][p1] = marker1
#    x[i][p2] = marker2

# segment sentences
seg1, seg2, seg3, seg4 = [], [], [], []
for i, instance in enumerate(instances[:sz]):
    sent = list(instance[0])
    p1, p2 = instance[1]
    # lstm add marker
    #sent[p1] = marker1
    #sent[p2] = marker2
    seg1.append(sent[:p1 + 1])
    seg2.append(sent[:p2 + 1])
    seg3.append(list(reversed(sent[p2:])))
    seg4.append(list(reversed(sent[p1:])))

l1, l2, l3, l4 = 15, 20, 20, 25
def norm_len(seg, length):
    if len(seg) > length:
        seg = seg[:length]
    else:
        seg += [padding] * (length - len(seg))
    return seg
for i in xrange(sz):
    seg1[i] = norm_len(seg1[i], l1)
    seg2[i] = norm_len(seg2[i], l2)
    seg3[i] = norm_len(seg3[i], l3)
    seg4[i] = norm_len(seg4[i], l4)
seg1 = np.array(seg1)
seg2 = np.array(seg2)
seg3 = np.array(seg3)
seg4 = np.array(seg4)

y = [item[2] for item in instances[:sz]]
# values = [1, 0, 3]
n_values = np.max(y) + 1
y = np.eye(n_values)[y]

# Randomly shuffle data
#np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]

seg1_shuffled = seg1[shuffle_indices]
seg2_shuffled = seg2[shuffle_indices]
seg3_shuffled = seg3[shuffle_indices]
seg4_shuffled = seg4[shuffle_indices]

# Split train/test set
# TODO: This is very crude, should use cross-validation
dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
seg1_train, seg1_dev = seg1_shuffled[:dev_sample_index], seg1_shuffled[dev_sample_index:]
seg2_train, seg2_dev = seg2_shuffled[:dev_sample_index], seg2_shuffled[dev_sample_index:]
seg3_train, seg3_dev = seg3_shuffled[:dev_sample_index], seg3_shuffled[dev_sample_index:]
seg4_train, seg4_dev = seg4_shuffled[:dev_sample_index], seg4_shuffled[dev_sample_index:]

print("Vocabulary Size: {:d}".format(vocabulary_size))
print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))


# Training
# ==================================================

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = TextCNN(
            B=FLAGS.batch_size,
            sequence_length=x_train.shape[1],
            sequence_length1=l1,
            sequence_length2=l2,
            sequence_length3=l3,
            sequence_length4=l4,
            num_classes=y_train.shape[1],
            vocab_size=vocabulary_size,
            embedding_size=FLAGS.embedding_dim,
            filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
            num_filters=FLAGS.num_filters,
            l2_reg_lambda=FLAGS.l2_reg_lambda)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.histogram_summary("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.scalar_summary("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.merge_summary(grad_summaries)

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.scalar_summary("loss", cnn.loss)
        acc_summary = tf.scalar_summary("accuracy", cnn.accuracy)

        # Train Summaries
        train_summary_op = tf.merge_summary([loss_summary, acc_summary, grad_summaries_merged])
        #train_summary_dir = os.path.join(out_dir, "summaries", "train")
        #train_summary_writer = tf.train.SummaryWriter(train_summary_dir, sess.graph)

        # Dev summaries
        dev_summary_op = tf.merge_summary([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.train.SummaryWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.all_variables())

        # Write vocabulary
        # vocab_processor.save(os.path.join(out_dir, "vocab"))

        # Initialize all variables
        sess.run(tf.initialize_all_variables())

        def train_step(x_batch, y_batch, seg1_batch, seg2_batch, seg3_batch, seg4_batch):
            """
            A single training step
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.input_seg1: seg1_batch,
              cnn.input_seg2: seg2_batch,
              cnn.input_seg3: seg3_batch,
              cnn.input_seg4: seg4_batch,
              cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
            }
            _, step, summaries, loss, accuracy = sess.run(
                [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            #time_str = datetime.datetime.now().isoformat()
            #if step % 20 == 0:
            #    print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            #train_summary_writer.add_summary(summaries, step)

        def dev_step(x_all, y_all, seg1_all, seg2_all, seg3_all, seg4_all, writer=None):
            """
            Evaluates model on a dev set
            """
            all_loss, all_acc = 0.0, 0.0
            num_batches = len(x_all) / FLAGS.batch_size
            for i in range(num_batches):
                start_index = i * FLAGS.batch_size
                end_index = min((i + 1) * FLAGS.batch_size, len(x_all))
                x_batch = x_all[start_index:end_index]
                y_batch = y_all[start_index:end_index]
                seg1_batch = seg1_all[start_index:end_index]
                seg2_batch = seg2_all[start_index:end_index]
                seg3_batch = seg3_all[start_index:end_index]
                seg4_batch = seg4_all[start_index:end_index]
                feed_dict = {
                  cnn.input_x: x_batch,
                  cnn.input_y: y_batch,
                  cnn.input_seg1: seg1_batch,
                  cnn.input_seg2: seg2_batch,
                  cnn.input_seg3: seg3_batch,
                  cnn.input_seg4: seg4_batch,
                  cnn.dropout_keep_prob: 1.0
                }
                step, summaries, loss, accuracy = sess.run(
                    [global_step, dev_summary_op, cnn.loss, cnn.accuracy], feed_dict)
                all_loss += loss
                all_acc += accuracy

            all_loss /= num_batches
            all_acc /= num_batches
            time_str = datetime.datetime.now().isoformat()

            # TODO: get W
            # with tf.variable_scope('embedding', reuse=True):
            #     W = tf.get_variable('W', [vocabulary_size+3, FLAGS.embedding_dim])
            #     print
            #     print 'W:', W[-3:, :10]
            #     print

            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, all_loss, all_acc))

            #if writer:
            #    writer.add_summary(summaries, step)

        # Generate batches
        batches = data_helpers.batch_iter(
            list(zip(x_train, y_train, seg1_train, seg2_train, seg3_train, seg4_train)),
            FLAGS.batch_size, FLAGS.num_epochs)
        # Training loop. For each batch...
        for batch in batches:
            x_batch, y_batch, x1_batch, x2_batch, x3_batch, x4_batch = zip(*batch)
            train_step(x_batch, y_batch, x1_batch, x2_batch, x3_batch, x4_batch)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.evaluate_every == 0:
                dev_step(x_train, y_train, seg1_train, seg2_train, seg3_train, seg4_train, writer=dev_summary_writer)
                dev_step(x_dev, y_dev, seg1_dev, seg2_dev, seg3_dev, seg4_dev, writer=dev_summary_writer)
            #if current_step % FLAGS.checkpoint_every == 0:
            #    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
            #    print("Saved model checkpoint to {}\n".format(path))
