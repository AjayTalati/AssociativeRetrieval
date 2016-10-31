import tensorflow as tf
import configuration

from LSTM_model import LSTM_model

def main():
  config = configuration.ModelConfig(data_filename="input_seqs_train")
  train(config)

def train(config):
  with tf.Graph().as_default():
    model = LSTM_model(config)
    inputs_seqs_batch, outputs_batch = model.reader.read()
    init_op = tf.group(tf.initialize_all_variables(),
                       tf.initialize_local_variables())

    sess = tf.Session()
    sess.run(init_op)
    saver = tf.train.Saver(tf.all_variables())
    global_steps = 0

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
      while not coord.should_stop():
        input_data, targets = sess.run([inputs_seqs_batch, outputs_batch])
        cost, _ = sess.run([model.cost, model.train_op], {model.input_data: input_data,
                                                          model.targets: targets})
        print("Step %d: cost:%f" % (global_steps,  cost))

        global_steps += 1
        if global_steps % 1000 == 0:
          model.inference(sess, inputs_seqs_batch, outputs_batch)
          print(saver.save(sess, "./save/LSTM/save", global_step=global_steps))
    except tf.errors.OutOfRangeError:
      print("Error")
    finally:
      # When done, ask the threads to stop.
      coord.request_stop()
    coord.join(threads)
    sess.close()

if __name__ == "__main__":
  main()