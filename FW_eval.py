"""
Fast weights evaluation part.
"""


import tensorflow as tf
import configuration
import numpy as np

from FW_model import FW_model

def main():
  config = configuration.ModelConfig(data_filename="input_seqs_eval")
  train(config)

def train(config):
  with tf.Graph().as_default():
    model = FW_model(config)
    inputs_seqs_batch, outputs_batch = model.reader.read(shuffle=False, num_epochs=1)
    init_op = tf.group(tf.initialize_all_variables(),
                       tf.initialize_local_variables())

    sess = tf.Session()
    sess.run(init_op)
    saver = tf.train.Saver(tf.all_variables())
    global_steps = 0

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    saver.restore(sess, "./save/FW/save-490000")

    correct_count = 0
    evaled_count = 0
    try:
      while not coord.should_stop():
        input_data, targets = sess.run([inputs_seqs_batch, outputs_batch])
        probs = sess.run([model.probs], {model.input_data: input_data,
                                                          model.targets: targets})
        probs = np.array(probs).reshape([-1, config.vocab_size])
        targets = np.array([t[0] for t in targets])
        output = np.argmax(probs, axis=1)

        correct_count += np.sum(output == targets)
        evaled_count += len(output)

    except tf.errors.OutOfRangeError:
        pass
    finally:
      # When done, ask the threads to stop.
      coord.request_stop()
    print("Accuracy: %f" % (float(correct_count) / evaled_count))
    coord.join(threads)
    sess.close()

if __name__ == "__main__":
  main()
