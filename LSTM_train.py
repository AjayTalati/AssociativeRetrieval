import tensorflow as tf
import configuration

from LSTM_model import LSTM_model

def main():
  config = configuration.ModelConfig()
  train(config)

def train(config):
  model = LSTM_model(config)
  with tf.Graph().as_default():
     inputs_seqs, outputs = model.reader.read()
     init_op = tf.group(tf.initialize_all_variables(),
                        tf.initialize_local_variables())

     sess = tf.Session()
     sess.run(init_op)

     coord = tf.train.Coordinator()
     threads = tf.train.start_queue_runners(sess=sess, coord=coord)

     try:
       while not coord.should_stop():
         inputs_seqs_batch, outputs_batch = sess.run([inputs_seqs, outputs])
         print(inputs_seqs_batch, outputs_batch)
     except tf.errors.OutOfRangeError:
       print("Error")
     finally:
       # When done, ask the threads to stop.
       coord.request_stop()
     coord.join(threads)
     sess.close()

if __name__ == "__main__":
  main()