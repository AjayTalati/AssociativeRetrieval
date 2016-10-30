class ModelConfig(object):

  def __init__(self, seq_length=4, rnn_size=20, batch_size=128, grad_clip=5.0, data_filename=None):
    self.seq_length = seq_length
    self.input_length = self.seq_length * 2 + 3
    self.rnn_size = rnn_size
    self.batch_size = batch_size
    self.grad_clip = grad_clip
    self.vocab_size = 37
    self.data_filename = data_filename

