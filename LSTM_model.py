import tensorflow as tf
from tensorflow.python.ops import rnn_cell
import utils

class LSTM_model(object):
  def __init__(self, config=None, mode=None):
    self.config = config
    self.mode = mode

    self.reader = utils.DataReader()
