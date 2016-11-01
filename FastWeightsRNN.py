from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest
from tensorflow.contrib.layers.python.layers import utils
from tensorflow.python.framework import ops
from tensorflow.python.ops import nn
from tensorflow.contrib.framework.python.ops import variables


def layer_norm(inputs,
               center=True,
               scale=True,
               activation_fn=None,
               reuse=None,
               variables_collections=None,
               outputs_collections=None,
               trainable=True,
               scope=None):
  """Adds a Layer Normalization layer from https://arxiv.org/abs/1607.06450.
    "Layer Normalization"
    Jimmy Lei Ba, Jamie Ryan Kiros, Geoffrey E. Hinton
  Can be used as a normalizer function for conv2d and fully_connected.
  Args:
    inputs: a tensor with 2 or more dimensions. The normalization
            occurs over all but the first dimension.
    center: If True, subtract `beta`. If False, `beta` is ignored.
    scale: If True, multiply by `gamma`. If False, `gamma` is
      not used. When the next layer is linear (also e.g. `nn.relu`), this can be
      disabled since the scaling can be done by the next layer.
    activation_fn: activation function, default set to None to skip it and
      maintain a linear activation.
    reuse: whether or not the layer and its variables should be reused. To be
      able to reuse the layer scope must be given.
    variables_collections: optional collections for the variables.
    outputs_collections: collections to add the outputs.
    trainable: If `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see tf.Variable).
    scope: Optional scope for `variable_op_scope`.
  Returns:
    A `Tensor` representing the output of the operation.
  Raises:
    ValueError: if rank or last dimension of `inputs` is undefined.
  """
  variable_scope = vs
  with variable_scope.variable_scope(scope, 'LayerNorm', [inputs],
                                     reuse=reuse) as sc:
    inputs = ops.convert_to_tensor(inputs)
    inputs_shape = inputs.get_shape()
    inputs_rank = inputs_shape.ndims
    if inputs_rank is None:
      raise ValueError('Inputs %s has undefined rank.' % inputs.name)
    dtype = inputs.dtype.base_dtype
    axis = list(range(1, inputs_rank))
    params_shape = inputs_shape[-1:]
    if not params_shape.is_fully_defined():
      raise ValueError('Inputs %s has undefined last dimension %s.' % (
          inputs.name, params_shape))
    # Allocate parameters for the beta and gamma of the normalization.
    beta, gamma = None, None
    if center:
      beta_collections = utils.get_variable_collections(variables_collections,
                                                        'beta')
      beta = variables.model_variable('beta',
                                      shape=params_shape,
                                      dtype=dtype,
                                      initializer=init_ops.zeros_initializer,
                                      collections=beta_collections,
                                      trainable=trainable)
    if scale:
      gamma_collections = utils.get_variable_collections(variables_collections,
                                                         'gamma')
      gamma = variables.model_variable('gamma',
                                       shape=params_shape,
                                       dtype=dtype,
                                       initializer=init_ops.ones_initializer,
                                       collections=gamma_collections,
                                       trainable=trainable)
    # Calculate the moments on the last axis (layer activations).
    mean, variance = nn.moments(inputs, axis, keep_dims=True)
    # Compute layer normalization using the batch_normalization function.
    variance_epsilon = 1E-12
    outputs = nn.batch_normalization(
        inputs, mean, variance, beta, gamma, variance_epsilon)
    outputs.set_shape(inputs_shape)
    if activation_fn is not None:
      outputs = activation_fn(outputs)
    return utils.collect_named_outputs(outputs_collections,
                                       sc.original_name_scope,
                                       outputs)


class LayerNormFastWeightsBasicRNNCell(rnn_cell.RNNCell):

  def __init__(self, num_units, forget_bias=1.0, reuse_norm=False,
               input_size=None, activation=nn_ops.relu,
               layer_norm=True, norm_gain=1.0, norm_shift=0.0,
               loop_steps=1, decay_rate=0.9, learning_rate=0.5,
               dropout_keep_prob=1.0, dropout_prob_seed=None):

    if input_size is not None:
      logging.warn("%s: The input_size parameter is deprecated.", self)

    self._num_units = num_units
    self._activation = activation
    self._forget_bias = forget_bias
    self._reuse_norm = reuse_norm
    self._keep_prob = dropout_keep_prob
    self._seed = dropout_prob_seed
    self._layer_norm = layer_norm
    self._S = loop_steps
    self._eta = learning_rate
    self._lambda = decay_rate
    self._g = norm_gain
    self._b = norm_shift

  @property
  def state_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._num_units

  def _norm(self, inp, scope=None):
    with vs.variable_scope(scope or "Norm") as scope:
      normalized = layer_norm(inp, reuse=self._reuse_norm, scope=scope)
      return normalized

  def _linear(self, args, output_size, bias, bias_start=0.0, scope=None):
    """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.
    Args:
      args: a 2D Tensor or a list of 2D, batch x n, Tensors.
      output_size: int, second dimension of W[i].
      bias: boolean, whether to add a bias term or not.
      bias_start: starting value to initialize the bias; 0 by default.
      scope: VariableScope for the created subgraph; defaults to "Linear".
    Returns:
      A 2D Tensor with shape [batch x output_size] equal to
      sum_i(args[i] * W[i]), where W[i]s are newly created matrices.
    Raises:
      ValueError: if some of the arguments has unspecified or wrong shape.
    """
    if args is None or (nest.is_sequence(args) and not args):
      raise ValueError("`args` must be specified")
    if not nest.is_sequence(args):
      args = [args]

    # Calculate the total size of arguments on dimension 1.
    total_arg_size = 0
    shapes = [a.get_shape().as_list() for a in args]
    for shape in shapes:
      if len(shape) != 2:
        raise ValueError("Linear is expecting 2D arguments: %s" % str(shapes))
      if not shape[1]:
        raise ValueError("Linear expects shape[1] of arguments: %s" % str(shapes))
      else:
        total_arg_size += shape[1]

    dtype = [a.dtype for a in args][0]

    # Now the computation. Matrix will be used multiple times in fast weights rnn, so reuse is True.
    with vs.variable_scope(scope or "Linear", reuse=True):
      matrix = vs.get_variable(
        "Matrix", [total_arg_size, output_size], dtype=dtype)
      if len(args) == 1:
        res = math_ops.matmul(args[0], matrix)
      else:
        res = math_ops.matmul(array_ops.concat(1, args), matrix)
      if not bias:
        return res
      bias_term = vs.get_variable(
        "Bias", [output_size],
        dtype=dtype,
        initializer=init_ops.constant_initializer(
          bias_start, dtype=dtype))
    return res + bias_term

  def __call__(self, inputs, state, scope=None):
    state, fast_weights = state
    with vs.variable_scope(scope or type(self).__name__) as scope:
      """Compute Wh(t)+Cx(t)"""
      linear = self._linear([inputs, state], self._num_units, False)
      """Compute h_0(t+1) = f(Wh(t)+Cx(t)"""
      if self._reuse_norm:
        h = self._activation(self._norm(linear, scope="Norm0"))
      else:
        h = self._activation(self._norm(linear))
      for i in range(self._S):
        """Compute h_{s+1}(t+1)=f([Wh(t)+Cx(t)]+A(t)h_s(t+1)) S times"""
        if self._reuse_norm:
          h = self._activation(self._norm(linear +
                                          math_ops.matmul(fast_weights, h), scope="Norm%d" % (i + 1)))
        else:
          h = self._activation(self._norm(linear +
                                          math_ops.matmul(fast_weights, h)))

      """Compute A(t+1)"""
      new_fast_weights = self._lambda * fast_weights + self._eta * math_ops.matmul(state, state, transpose_a=True)

      return h, new_fast_weights