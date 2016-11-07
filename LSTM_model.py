import tensorflow as tf
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import seq2seq
import utils

import numpy as np

class LSTM_model(object):
  def __init__(self, config=None, mode=None):
    self.config = config
    self.mode = mode
    self.build_graph()
    self.load_validation()

  def load_validation(self):
    data_reader = utils.DataReader(data_filename="input_seqs_validation", batch_size=16)
    inputs_seqs_batch, outputs_batch = data_reader.read(False, 1)
    init_op = tf.group(tf.initialize_all_variables(),
                       tf.initialize_local_variables())

    sess = tf.Session()
    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    self.validation_inputs = []
    self.validation_targets = []
    try:
      while not coord.should_stop():
        input_data, targets = sess.run([inputs_seqs_batch, outputs_batch])
        self.validation_inputs.append(input_data)
        self.validation_targets.append(targets)
    except tf.errors.OutOfRangeError:
      pass
    finally:
      coord.request_stop()
    coord.join(threads)
    sess.close()

    self.validation_inputs = np.array(self.validation_inputs).reshape([-1, self.config.input_length])
    self.validation_targets = np.array(self.validation_targets).reshape([-1, 1])

  def build_graph(self):
    config = self.config
    self.reader = utils.DataReader(seq_len=config.seq_length, batch_size=config.batch_size, data_filename=config.data_filename)

    self.cell = rnn_cell.BasicLSTMCell(config.rnn_size, state_is_tuple=True)

    self.input_data = tf.placeholder(tf.int32, [None, config.input_length])
    self.targets = tf.placeholder(tf.int32, [None, 1])
    self.initial_state = self.cell.zero_state(tf.shape(self.targets)[0], tf.float32)

    with tf.variable_scope("input_embedding"):
      embedding = tf.get_variable("embedding", [config.vocab_size, config.rnn_size])
      inputs = tf.split(1, config.input_length, tf.nn.embedding_lookup(embedding, self.input_data))
      inputs = [tf.squeeze(input, [1]) for input in inputs]

    with tf.variable_scope("send_to_rnn"):
      state = self.initial_state
      output = None

      for i, input in enumerate(inputs):
        if i > 0:
          tf.get_variable_scope().reuse_variables()
        output, state = self.cell(input, state)

    with tf.variable_scope("softmax"):
      softmax_w = tf.get_variable("softmax_w", [config.rnn_size, config.vocab_size])
      softmax_b = tf.get_variable("softmax_b", [config.vocab_size])
      self.logits = tf.matmul(output, softmax_w) + softmax_b
      self.probs = tf.nn.softmax(self.logits)
      self.output = tf.cast(tf.reshape(tf.arg_max(self.probs, 1), [-1, 1]), tf.int32)
      self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.output, self.targets), tf.float32))

    loss = seq2seq.sequence_loss_by_example([self.logits],
                                            [tf.reshape(self.targets, [-1])],
                                            [tf.ones([config.batch_size])],
                                            config.vocab_size)

    self.cost = tf.reduce_mean(loss)
    self.final_state = state

    # self.lr = tf.Variable(0.001, trainable=False)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars),
                                      config.grad_clip)
    optimizer = tf.train.AdamOptimizer()#self.lr)
    self.train_op = optimizer.apply_gradients(zip(grads, tvars))

    self.summary_accuracy = tf.scalar_summary('accuracy', self.accuracy)
    tf.scalar_summary('cost', self.cost)
    self.summary_all = tf.merge_all_summaries()
