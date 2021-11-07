# ADDED by Jannis Born - starting on 21 April 2018
# - Customized LdS sequence loss function

# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import script_ops
from tensorflow.python.client import session
from .lds_utils import update_tensor

import numpy as np
import warnings, os
warnings.filterwarnings("ignore",category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 




__all__ = ["sequence_loss_lds"]


def sequence_loss_lds(logits,
                  targets,
                  weights,
                  alt_targets,
                  max_alt_spellings,
                  print_ratio=True,
                  name='sequence_loss_lds'):
  """  
  Modified sequence_loss function for the "Lesen durch Schreiben" project

  Modifications include:
  1)  Handing over ALT_TARGETS, a np.array of alternative writings for every word
  2)  Functioning: softmax output is computed regularly. Then, predictions are gen-
        erated (tf.argmax) and is compared to the true targets and the alt. targets.
        If a matching was found, logits are manually set to the right sequence with p=1


  Parameters:
  ------------
  LOGITS      {tf.Tensor} of shape [batch_size, sequence_length, num_decoder_symbols] 
                  and dtype float32. Corresponds to the prediction across all classes at
                  each timestep. If SOFTMAX is True, LOGITS are ]-inf, inf[ if SOFTMAX
                  is False, LOGITS are indeed softmax outputs [0,1].
  TARGETS     {tf.Tensor} of shape [batch_size, sequence_length] and dtype int64. TARGETS
                  represent the true orthographic spelling of every word.
  WEIGHTS     {tf.Tensor} of shape [batch_size, sequence_length] and dtype float. 
                  WEIGHTS constitutes the weighting of each prediction in the sequence. 
                  For my case: set all values to 1 (set 0 to skip prediction)
  ALT_TARGETS {tf.Tensor} of shape [batch_size, sequence_length, max_alt_spellings] and
                  dtype int32. ALT_TARGETS represent the alternative orthographic spellings
                  that are also considered as being "correct" by the teacher. 
                  max_alt_spellings refers to the amount of alternative "correct" spellings
                  of the word with the most alternative correct spellings. If a word has < 
                  max_alt_spellings the rest of the 3. dim. should be filled with float('NaN')
  PRINT_RATIO {bool}, Whether the ratio of words that were orthographically incorrect, but accepted 
                  from LdS teacher should be printed.
  NAME        {str}, optional. Name for this operation, defaults to "sequence_loss_lds"


  Returns:
  ------------
  CROSSENT    {tf.float32} Tensor of rank 1, carrying the cross entropy.
  FIN_TARGETS {tf.Tensor} of shape batch_size x seq_len 


  Raises:
    ValueError: logits does not have 3 dimensions or targets does not have 2
                dimensions or weights does not have 2 dimensions.
  """

  # Error handling
  if len(logits.get_shape()) != 3:
    raise ValueError("Logits must be a [batch_size x sequence_length x logits] tensor")
  if len(targets.get_shape()) != 2:
    raise ValueError("Targets must be a [batch_size x sequence_length] tensor")
  if len(weights.get_shape()) != 2:
    raise ValueError("Weights must be a [batch_size x sequence_length] tensor")


  with ops.name_scope(name, "sequence_loss_lds", [logits, targets, alt_targets, weights, print_ratio,max_alt_spellings]):



      batch_size = targets.get_shape()[0]
      seq_len = targets.get_shape()[1]
      num_classes = logits.get_shape()[2]
      logits_flat = array_ops.reshape(logits, [-1, num_classes]) # Pseudoflat logits of shape [batch_size*seq_len x num_classes]
      
      
      # IDEA:
      # For every output word of the batch, check whether it matches to any alternative writing
      # If so, replace the target by the alternative writing, else take original target.

      # For loops are a hustle in TF, thus we make an elementwise compparison
      writings = math_ops.cast(math_ops.argmax(logits,axis=-1),dtypes.int32)


      # wr is now max_alt_writings x bs x seq_len with the first repeated max_alt_writing times
      # writings_all is intended to make a comparison with an array that includes REAL target, writings leaves that out.
      writings_all = array_ops.expand_dims(array_ops.ones([max_alt_spellings+1,1],dtype=dtypes.int32), 1) * writings
      writings     = array_ops.expand_dims(array_ops.ones([max_alt_spellings,1],dtype=dtypes.int32), 1) * writings
      # transpose to prepare for elementwise comparison
      writings_all = array_ops.transpose(writings_all,[1,2,0])
      writings = array_ops.transpose(writings,[1,2,0])
      # perform elementwise comparison. Result of shape bs x seq_len x max_alt_writings
      equal_raw = gen_math_ops.equal(writings,alt_targets)
      # collapse the seq_len dimension, hape gets bs x max_alt_writings, s.t.row-wise the indices of True hold which
      # writing was generated.
      equal = math_ops.reduce_all(equal_raw,1) 
      # Get these indices. Result is shape ? x 2 but ?=bs if every word was written acc. to 1! alt. writing 
      equal_ind = array_ops.where(equal)


      # Compute ratio of correctly spelled words, concatenate true and lds-true spellings
      both_targs = array_ops.concat([alt_targets,array_ops.expand_dims(targets,2)],2)
      equal_both_ind = array_ops.where(math_ops.reduce_all(gen_math_ops.equal(writings_all,both_targs),1))

      # Outsource the target tensor update operation from TF to python/numpy (excluded from tensor graph)
      new_targets, ratio_lds, ratio_corr  = script_ops.py_func(update_tensor,[targets, alt_targets, equal_ind, equal_both_ind], 
        [dtypes.int64, dtypes.float64, dtypes.float64])


      crossent = nn_ops.sparse_softmax_cross_entropy_with_logits(labels=array_ops.reshape(new_targets, [-1]), logits=logits_flat)
      crossent *= array_ops.reshape(weights, [-1])
      crossent = math_ops.reduce_sum(crossent)
      total_size = math_ops.reduce_sum(weights)
      total_size += 1e-12 
      crossent /= total_size

      crossent_reg = nn_ops.sparse_softmax_cross_entropy_with_logits(labels=array_ops.reshape(targets, [-1]), logits=logits_flat)
      crossent_reg *= array_ops.reshape(weights, [-1])
      crossent_reg = math_ops.reduce_sum(crossent_reg)
      crossent_reg /= total_size

      return crossent, new_targets, ratio_lds, ratio_corr, crossent_reg

