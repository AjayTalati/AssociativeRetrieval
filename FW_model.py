import tensorflow as tf
from FastWeightsRNN import LayerNormFastWeightsBasicRNNCell
from tensorflow.python.ops import seq2seq
import utils

import numpy as np

class FW_model(object):
  def __init__(self, config=None, mode=None):
    self.config = config
    self.mode = mode

    self.reader = utils.DataReader(seq_len=config.seq_length, batch_size=config.batch_size, data_filename=config.data_filename)

    self.cell = LayerNormFastWeightsBasicRNNCell(num_units=config.rnn_size)

    self.input_data = tf.placeholder(tf.int32, [None, config.input_length])
    self.targets = tf.placeholder(tf.int32, [None, 1])
    self.initial_state = self.cell.zero_state(tf.shape(self.targets)[0], tf.float32)
    self.initial_fast_weights = self.cell.zero_fast_weights(tf.shape(self.targets)[0], tf.float32)

    with tf.variable_scope("input_embedding"):
      embedding = tf.get_variable("embedding", [config.vocab_size, config.rnn_size])
      inputs = tf.split(1, config.input_length, tf.nn.embedding_lookup(embedding, self.input_data))
      inputs = [tf.squeeze(input, [1]) for input in inputs]

    with tf.variable_scope("send_to_rnn"):
      state = (self.initial_state, self.initial_fast_weights)
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
    optimizer = tf.train.AdamOptimizer()  # self.lr)
    self.train_op = optimizer.apply_gradients(zip(grads, tvars))

  def inference(self, sess, input_data, targets):
    input_data, targets = sess.run([input_data, targets])
    probs = sess.run(self.probs, {self.input_data: input_data,
                                  self.targets: targets})
    output = np.argmax(probs, axis=1)
    targets = np.array([t[0] for t in targets])
    print("Accuracy: ", np.sum(output == targets) / float(len(targets)))
