class ModelConfig(object):

  def __init__(self):
    self.seq_length = 4
    self.input_length = self.seq_length * 2 + 3
    self.rnn_size = 20
    self.batch_size = 128
    self.grad_clip = 5.0
    self.vocab_size = 37

