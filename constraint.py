# Implementation of weight clipping constraint

import tensorflow as tf

class MinMax(tf.keras.constraints.Constraint):
  """
  Constrains weight tensors to be between ref_range[0] and ref_range[1]`.
  """

  def __init__(self, ref_range):
    self.ref_range = ref_range

  def __call__(self, w):
    w=tf.math.maximum(w, self.ref_range[0])
    w=tf.math.minimum(w, self.ref_range[1])
    return w

  def get_config(self):
    return {'ref_range': self.ref_range}