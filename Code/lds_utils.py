
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
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.client import session

import numpy as np
import warnings, os
warnings.filterwarnings("ignore",category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'




def update_tensor(targets, alt_targets, equal_inds, equal_both_inds):
    """
    Wrapper for tf.py_func. 
    Expects input arguments to be numpy arrays, but the py_func wrapper can be called directly from the tf graph via tensors.
    
    Parameters:
    -------------
    TARGETS         {np.array} of shape batch_size x sequence_length, the conventional labels
    ALT_TARGETS     {np.array} of shape batch_size x sequence_length x max_alt_writings, the list of alternative, correct spellings
    EQUAL_INDS      {np.array} of shape num_lds_spellings x 2, with one row per word that was written in a "LdS-correct sense"
                        First column gives index of word in batch, second gives index of spelling in max_alt_writings 

    Returns:
    -------------
    NEW_TARGETS     {np.array} of shape batch_size x sequence_length, the updated targets
/
    """

    dict_in = {1: 't', 2: 'k', 3: 'I', 4: 'j', 5: 'g', 6: 'e', 7: 's', 8: '@', 9: 'E', 10: '#', 11: '|', 12: 'i', 13: 'Y', 14: 'l', 15: 'n', 16: 'f', 17: ':', 18: ' ', 19: 'U', 20: 'd', 21: 'u', 22: 'h', 23: 'S', 24: 'r', 25: 'v', 26: 'y', 27: 'o', 28: '/', 29: 'N', 30: 'p', 31: 'a', 32: 'x', 33: 'O', 34: 'z', 35: '+', 36: 'm', 37: 'b', 38: '<GO>', 39: '<PAD>'}
    dict_out = {1: 'c', 2: 't', 3: 'k', 4: 'j', 5: 'g', 6: 'e', 7: 's', 8: 'i', 9: 'l', 10: 'n', 11: 'f', 12: 'q', 13: ' ', 14: 'd', 15: 'u', 16: 'w', 17: 'h', 18: 'r', 19: 'v', 20: 'y', 21: 'o', 22: 'p', 23: 'a', 24: 'x', 25: 'z', 26: 'm', 27: 'b', 28: '<GO>', 29: '<PAD>'}

    bs = targets.shape[0]
    seq_len = targets.shape[1]
    
    which_lds_correct = equal_inds[:,0] # those words from the batch that had a lds-correct-spelling
    which_spelling_inds = equal_inds[:,1] # the index referring WHICH alternative wriitng was produced
    
    new_targets = np.zeros(targets.shape,dtype=np.int64)
    alt_targets = alt_targets.astype(np.int32,copy=False)

    for batch_ind in range(bs):
        
        # If the current word was spelled correctly in LdS sense
        if batch_ind in which_lds_correct:

            word_ind = which_spelling_inds[np.where(which_lds_correct==batch_ind)]
            word_ind = word_ind[0]
            new_targets[batch_ind,:] = alt_targets[batch_ind,:,word_ind]


            # Print alternative writings
            #label_str = ''.join([dict_out[l] if dict_out[l] != '<PAD>' and  dict_out[l] != '<GO>' else '' for l in targets[batch_ind,:]])
            #alt_label_str = ''.join([dict_out[l] if dict_out[l] != '<PAD>' and  dict_out[l] != '<GO>' else '' for l in alt_targets[batch_ind,:,word_ind]])
            #print(" LDS LOSS - The output " + alt_label_str.upper() + " was accepted for " + label_str.upper())
        
        # Otherwise just use the normal target
        else:
            new_targets[batch_ind,:] = targets[batch_ind,:]

    ratio_lds = len(which_lds_correct)/bs
    #print("LDS-LOSS - Ratio of correct words is ", str(ratio_lds))
    ratio_correct = (len(equal_both_inds)-len(which_lds_correct))/bs

    return new_targets, ratio_lds, ratio_correct
 