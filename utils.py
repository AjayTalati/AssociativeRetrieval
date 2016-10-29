from string import ascii_lowercase
from string import digits
import numpy as np
import tensorflow as tf
import os


class DataGenerator():
  def __init__(self, seq_len=10, batch_size=32, num_batch=10, data_directory="./data", data_filename="input_seq"):
    self.seq_len = seq_len
    self.batch_size = batch_size
    self.num_batch = num_batch
    self.data_directory = data_directory
    self.data_filename = data_filename
    self.vocab_list = self.generate_vocab_list()
    self.vocab_dict = self.generate_vocab_dict()
    self.decode_dict = self.generate_decode_dict()

  def generate_vocab_list(self):
    return [c for c in (ascii_lowercase + digits + "?")]

  def generate_vocab_dict(self):
    return dict(zip(self.vocab_list, range(len(self.vocab_list))))

  def generate_decode_dict(self):
    return dict(zip(range(len(self.vocab_list)), self.vocab_list))

  def generate_input_sequence(self):
    ascii_list = [c for c in ascii_lowercase]
    digit_list = [c for c in digits]

    sample_char_seq = np.random.choice(ascii_list, self.seq_len, replace=False).tolist()
    sample_digit_seq = np.random.choice(digit_list, self.seq_len, replace=False).tolist()

    query_char = np.random.choice(sample_char_seq, 1).tolist()[0]
    query_result = sample_digit_seq[sample_char_seq.index(query_char)]

    output = []
    for pair in zip(sample_char_seq, sample_digit_seq):
      output += list(pair)

    output += ['?'] * 2
    output += [query_char]

    return (list(map(lambda x: self.vocab_dict[x], output)), [self.vocab_dict[query_result]])

  def generate_inputs(self):
    inputs = []
    for _ in range(self.num_batch):
      for _ in range(self.batch_size):
        inputs.append(self.generate_input_sequence())

    return inputs

  def _int64_feature(self, value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

  def generate_inputs_file(self):
    inputs = self.generate_inputs()

    if not os.path.exists(self.data_directory):
      os.mkdir(self.data_directory)

    filename = os.path.join(self.data_directory, self.data_filename + ".tfrecords")
    writer = tf.python_io.TFRecordWriter(filename)

    for item in inputs:
      example = tf.train.Example(features=tf.train.Features(feature={
        'inputs_seq': self._int64_feature(item[0]),
        'output': self._int64_feature(item[1])
      }))
      writer.write(example.SerializeToString())

class DataReader():
  def __init__(self, data_directory="./data", data_filename="input_seq"):
    self.filename_queue = tf.train.string_input_producer([os.path.join(data_directory, data_filename + ".tfrecords")])

  def read(self):
    reader = tf.TFRecordReader()
    _, serialized_input = reader.read(self.filename_queue)
    inputs = tf.parse_single_example(serialized_input,
                                      features={
                                        'inputs_seq': tf.FixedLenFeature([23], tf.int64),
                                        'output': tf.FixedLenFeature([1], tf.int64)
                                        #'inputs_seq': tf.VarLenFeature(tf.int64),
                                        #'output': tf.VarLenFeature(tf.int64)
                                      })
    inputs_seq = inputs['inputs_seq']
    output = inputs['output']
    inputs_seqs, outputs = tf.train.shuffle_batch([inputs_seq, output], batch_size=2, num_threads=1,capacity=16, min_after_dequeue=10)
    return inputs_seqs, outputs

if __name__ == "__main__":
  # gen = DataGenerator()
  # print(gen.generate_vocab_dict())
  # print(gen.generate_input_sequence())
  # print(gen.generate_inputs())
  # gen.generate_inputs_file()
  reader = DataReader()
  print("read:\n\n")

  sess = tf.Session()
  init = tf.initialize_all_variables()
  tf.train.start_queue_runners(sess)
  outputs = (sess.run(reader.read()))
  # print (outputs)
  a, b = reader.read()
  a_np, b_np = sess.run([a, b])
  print(a_np, b_np)
