import warnings, os, sys
warnings.filterwarnings("ignore",category=FutureWarning)
import pickle
import tensorflow as tf
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 


def batch_data(x, y, BATCH_SIZE, alt_targs=None):
    """
    Receives a batch_size and the entire training data [i.e inputs (x) and labels (y)]
    Returns a data iterator
    """

    shuffle = np.random.permutation(len(x))
    start = 0
    x = x[shuffle]
    y = y[shuffle]

    if alt_targs is None:

        while start + BATCH_SIZE <= len(x):
            yield x[start:start+BATCH_SIZE], y[start:start+BATCH_SIZE]
            start += BATCH_SIZE

    else:
        alt_targs = alt_targs[shuffle]
        while start + BATCH_SIZE <= len(x):
            yield x[start:start+BATCH_SIZE], y[start:start+BATCH_SIZE], alt_targs[start:start+BATCH_SIZE]
            start += BATCH_SIZE

   


def accuracy_prepare(logits, labels, char2numY, mode='train'):
    """ Method to prepare logits and labels for accuracy evaluation by removing padding values """

    # Error handling and mode setting
    if mode =='train':
        fullPred = logits.argmax(-1) # Prediction string with padding
    elif mode == 'test':
        fullPred = np.copy(logits.astype(int))
    else:
        print("Please specify 'mode' as either 'train' or 'test'. ")
    
    #Padded target string
    fullTarg = np.copy(labels) 
    # Set pads to 0 - as preparation for edit_distance
    if '<PAD>' in char2numY:
        fullPred[fullPred==char2numY['<PAD>']] = 0
        fullTarg[fullTarg==char2numY['<PAD>']] = 0

    return fullPred, fullTarg



class acc_new(object):

    def __init__(self):

        self.fullPred = tf.placeholder(tf.int64,(None,None))
        self.fullTarg = tf.placeholder(tf.int64,(None,None))

    def accuracy(self):

        self.dists = tf.edit_distance(self.dense_to_sparse(self.fullPred), self.dense_to_sparse(self.fullTarg))
        self.token_acc = 1 - tf.reduce_mean(self.dists)

    def dense_to_sparse(self,denseTensor):

        # Non-Zero indices of dense tensor
        idx = tf.where(tf.not_equal(denseTensor,0))
        # Create sparse tensor
        sparseTensor = tf.SparseTensor(idx, tf.gather_nd(denseTensor,idx), tf.cast(tf.shape(denseTensor),tf.int64))
        return sparseTensor

def np_dict_to_dict(np_dict):
    """ 
    Converts a dictionary saved via np.save (as structured np array) into an object of type dict
    Parameters:         NP_DICT        : {np.array} structured np.array with dict keys and items
    Returns:            DICT           : {dict} converted NP_DICT
    """
    return {key:np_dict.item().get(key) for key in np_dict.item()}


def set_model_params(inputs, targets, dict_char2num_x, dict_char2num_y):
    """
    This method can receive data from any dataset (inputs, targets) and the corresponding dictionaries.
    It returns the hyperparameters for the model, i.e. input and output sequence length as well as input and output dictionary size.
    """

    # Error handling. If the dicts are not objects of type dict but np.arrays (dicts saved via np.save), convert them back.
    if isinstance(dict_char2num_x, np.ndarray):
        dict_char2num_x = np_dict_to_dict(dict_char2num_x)
    if isinstance(dict_char2num_y, np.ndarray):
        dict_char2num_y = np_dict_to_dict(dict_char2num_y)


    dict_num2char_x = dict(zip(dict_char2num_x.values(), dict_char2num_x.keys()))
    dict_num2char_y = dict(zip(dict_char2num_y.values(), dict_char2num_y.keys()))
    x_dict_size = len(dict_char2num_x)
    num_classes = len(dict_char2num_y) # (y_dict_size) Cardinality of output dictionary
    x_seq_length = len(inputs[0]) - 1 
    y_seq_length = len(targets[0]) - 1 # Because of the <GO> as response onset

    return x_dict_size, num_classes, x_seq_length, y_seq_length, dict_num2char_x, dict_num2char_y



def extract_celex(path):
    """
    Reads in data from the CELEX corpus
    
    Parameters:
    -----------
    PATH        {str} the path to the desired celex file, i.e. gpl.cd 
                    (contains orthography and phonology)

    Returns:
    -----------
    2 Tuples, each with 2 variables. 
        First tuple:
    W           {np.array} of words (length 51728) for gpl.cd
    P           {np.array} of phoneme sequences (length 51728) for gpl.cd
        Second tuple:
    WORD_DICT   {dict} allowing to map the numerical array W back to strings
    PHON_DICT   {dict} doing the same for the phonetical arrays P

    
    Call via:
    ((w,p) , (word_dict, phon_dict)) = extract_celex(path)
    
    """
    
    
    with open(path, 'r') as file:

        raw_data = file.read().splitlines()
        words = []
        phons = []
        m = 0
        t = 0
        for ind,raw_line in enumerate(raw_data):
            
            line = raw_line.split("\\")

            if line[-2]: # Use only words that HAVE a SAMPA transcript (reduces from 51k to 37345)

            # exclude foreign words that have the 'æ' tone (SAMPA '{' ) like in PoINte   - 18 words
            # exclude foreign words that have the 'ɑ' tone (SAMPA 'A' ) like in NuANce   - 28 words
            # exclude foreign words that have a nasal vowel (SAMPA '~' ) like in Jargon  - 22 words
                if not 'A' in line[-2] and not '{' in line[-2] and not '~' in line[-2]: 

                    if not ('tS' in line[-2] and not 'tsch' in line[1]): # exclude 9 foreign words like 'Image', 'Match', 'Punch', 'Sketch'
                        
                        if not ('e' in line[-2] and not 'eː' in line[-2]): # exclude aerosol

                            if len(line[1]) < 10 and len(line[-2]) < 10 : # exclude extra long words 

                                if len(line[-2]) > m:
                                    m = len(line[-2])
                                    print(line[1],line[-2])


                                words.append(line[1].lower()) # All words are lowercase only
                                phons.append(line[-2]) # Using SAMPA notation
                            
                        else:
                            t+=1
                            
    print("Excluded",t, "words because they were too long (more than 15 phons)" )
    print("Size of dataset is", len(words), "samples")

    return words,phons


 
def celex_retrieve(path: str):
    """
    Retrives the previously saved data from the CELEX corpus
    """

    data = np.load(os.path.join(path, 'celex_few_lds.npz'))
    phon_dict = np_dict_to_dict(data['phon_dict'])
    word_dict = np_dict_to_dict(data['word_dict'])

    print("Loading alternative targets ...")
    alt_targs_raw = np.load(os.path.join(path, 'celex_few_lds_alt_targets.npy'))

    alt_targs = np.array([np.array(d,dtype=np.int8) for d in alt_targs_raw])
    print("Alternative targets successfully loaded.")

    return ( (data['phons'], data['words']) , (phon_dict, word_dict), alt_targs )


def celex_all_retrieve(path: str):
    """
    Retrives the previously saved data from the CELEX corpus
    """

    data = np.load(os.path.join(path, 'celex_all.npz'))
    phon_dict = np_dict_to_dict(data['phon_dict'])
    word_dict = np_dict_to_dict(data['word_dict'])

    print("Loading alternative targets ...")
    alt_targs_raw = np.load(os.path.join(path, 'celex_all_alt_targets.npy'))

    alt_targs = np.array([np.array(d,dtype=np.int8) for d in alt_targs_raw])
    print("Alternative targets successfully loaded.")

    return ( (data['phons'], data['words']) , (phon_dict, word_dict), alt_targs )


def childlex_retrieve(path: str):
    """
    Retrives the previously saved data from the childlex database (subset of CELEX)
    """

    data = np.load(os.path.join(path, 'childlex.npz'))
    phon_dict = np_dict_to_dict(data['phon_dict'])
    word_dict = np_dict_to_dict(data['word_dict'])

    print("Loading alternative targets ...")
    alt_targs_raw = np.load(os.path.join(path, 'childlex_alt_targets.npy'))


    alt_targs = np.array([np.array(d,dtype=np.int8) for d in alt_targs_raw])
    print("Alternative targets successfully loaded.")

    return ( (data['phons'], data['words']) , (phon_dict, word_dict), alt_targs )


def fibel_retrieve(path:str):

    data = np.load(os.path.join(path, 'fibel.npz'))
    phon_dict = np_dict_to_dict(data['phon_dict'])
    word_dict = np_dict_to_dict(data['word_dict'])


    print("Loading alternative targets ...")
    alt_targs_raw = np.load(os.path.join(path, 'fibel_alt_targets.npy'))


    alt_targs = np.array([np.array(d,dtype=np.int8) for d in alt_targs_raw])
    print("Alternative targets successfully loaded.")

    return ( (data['phons'], data['words']) , (phon_dict, word_dict), alt_targs )
 

def get_last_id(path:str, dataset:str):

    # Retrieves the maximal ID of all the saved models in the path

    from os.path import expanduser
    
    path = os.path.join(path, dataset)

    try:
        folders = next(os.walk(path))[1]
        IDs = [int(s) for folder in folders for s in folder.split('_') if s.isdigit()]

    except StopIteration:
        IDs = [6,7]
        pass

    return max(IDs)


def lds_compare(logits, targets, alt_targets, dict_out, mode):
    """
    Like in sequence_loss_lds this method checks whether the generated predictions match any of the alternative targets

    Parameters:
    --------------
    LOGITS          {np.array}  {2D,3D}  of shape {batch_size x seq_len, bs x sl x num_classes} depending on whether mode is train or test 
    TARGETS         {np.array}  2D  of shape batch_size x seq_len
    ALT_TARGETS     {np.array}  3D  of shape batch_size x seq_len x max_alt_writings
    MODE            {string} from {'train','test'}

    Returns:
    --------------
    NEW_TARGETS     {np.array}  2D  of shape batch_size x seq_len

    """

    if mode == 'train':
        prediction = logits.argmax(axis=-1)
    elif mode == 'test':
        prediction = logits

    batch_size = targets.shape[0]
    max_alt_spellings = alt_targets.shape[2]

    new_targets = np.zeros(targets.shape, np.int64)
    counter = []

    for wo_ind in range(batch_size):
        wrote_alternative = False

        # Check whether the word was actually correctly spelled.
        if np.array_equal(prediction[wo_ind,:],targets[wo_ind,:]):
            new_targets[wo_ind,:] = targets[wo_ind,:]
            counter.append(wrote_alternative)

        else:
            # If not, check all the alternative writings
            for tar_ind in range(max_alt_spellings):
                if np.array_equal(prediction[wo_ind,:], alt_targets[wo_ind,:,tar_ind]):
                    wrote_alternative = True 
                    new_targets[wo_ind,:] = alt_targets[wo_ind,:,tar_ind]

                    # Print alternative writings
                    #out_str = ''.join([dict_out[l] if l!= 0 and dict_out[l] != '<PAD>' and  dict_out[l] != '<GO>' else '' for l in prediction[wo_ind,:]])
                    #label_str = ''.join([dict_out[l] if dict_out[l] != '<PAD>' and  dict_out[l] != '<GO>' else '' for l in targets[wo_ind,:]])
                    #alt_label_str = ''.join([dict_out[l] if dict_out[l] != '<PAD>' and  dict_out[l] != '<GO>' else '' for l in alt_targets[wo_ind,:,tar_ind]])
                    #print("UTILS - The output " + out_str.upper() + " was accepted for " + label_str.upper()+", alt. writing: "+alt_label_str.upper())
                    counter.append(wrote_alternative)

                    continue

            # In case the spelling was actuall bullshit
            if not wrote_alternative:
                new_targets[wo_ind,:] = targets[wo_ind,:]
                counter.append(wrote_alternative)
    
    # How many words were written alternatively?
    rat = sum(counter) / len(counter)
    #if rat > 0:
    #      print("UTILS - Ratio of words that were 'correct' in LdS sense: ", str(rat))

    return new_targets if mode == 'train' else new_targets, rat

def num_to_str(inputs,logits,labels,alt_targs,dict_in,dict_out):
    """
    Method receives the numerical arrays and prints the strings

    If mode = normal, then alt_targs should be set to [], if it is 'lds' the alternative targets are also printed
    """

    fullPred = logits.argmax(-1) # Prediction string with padding

    out_str = []
    inp_str = []
    label_str = []
    r=0
    for k in range(len(inputs)):
        out_str.append(''.join([dict_out[l] if l!= 0 and dict_out[l] != '<PAD>' and  dict_out[l] != '<GO>' else '' for l in fullPred[k]]))
        inp_str.append(''.join([dict_in[l] if dict_in[l] != '<PAD>' and  dict_in[l] != '<GO>' else '' for l in inputs[k]]))
        label_str.append(''.join([dict_out[l] if dict_out[l] != '<PAD>' and  dict_out[l] != '<GO>' else '' for l in labels[k]]))

        if k==8:
            print("The input " + inp_str[-1].upper() + " was written " + out_str[-1].upper() + " with target " + label_str[-1].upper())

        
        #if mode == 'lds':
        alt_targ_str = []

        z = np.argwhere(alt_targs[k]==0) # indices of zeros (alt_targs where padded to have equally sized array)
        # position 0,1 is the first row that contains zeros (i.e. not an alternative writing anymore)
        try:
            num_wrt = z[0,1]
        except IndexError: # for the only word with no zero rows
            num_wrt = len(alt_targs[k])

        for l in range(num_wrt):
            alt_targ_str.append(''.join([dict_out[m] if dict_out[m] != '<PAD>' and  dict_out[m] != '<GO>' else '' for m in alt_targs[k,:,l] ]))
        #print("The alternatives were ", alt_targ_str)

        if out_str[-1] in alt_targ_str:
            print("                    HERE!!!", out_str[-1],' instead of ', label_str[-1])
            r = k

    return r



def num_to_str_help(inputs,logits,labels,dict_in,dict_out):
    """
    Method receives the numerical arrays and prints the strings

    If mode = normal, then alt_targs should be set to [], if it is 'lds' the alternative targets are also printed
    """

    fullPred = logits# Prediction string with padding

    out_str = []
    inp_str = []
    label_str = []
    r=0
    for k in range(len(inputs)):
        out_str.append(''.join([dict_out[l] if l!= 0 and dict_out[l] != '<PAD>' and  dict_out[l] != '<GO>' else '' for l in fullPred[k]]))
        inp_str.append(''.join([dict_in[l] if dict_in[l] != '<PAD>' and  dict_in[l] != '<GO>' else '' for l in inputs[k]]))
        label_str.append(''.join([dict_out[l] if dict_out[l] != '<PAD>' and  dict_out[l] != '<GO>' else '' for l in labels[k]]))

        if k==2:
            print("The input " + inp_str[-1].upper() + " was written " + out_str[-1].upper() + " with target " + label_str[-1].upper())

        
        #if mode == 'lds':
        alt_targ_str = []
    return 42


def comp_reading(new_input,real_input,dict_word):

    for k in range(len(new_input)):

        if any(new_input[k,:] != real_input[k,:]):

            real_word = ''.join([dict_word[l] if l!= 0 and dict_word[l] != '<PAD>' and  dict_word[l] != '<GO>' else '' for l in real_input[k]])
            new_word = ''.join([dict_word[l] if l!= 0 and dict_word[l] != '<PAD>' and  dict_word[l] != '<GO>' else '' for l in new_input[k]])

            print("Instead of the word ", real_word," the word ", new_word, " is read.")




               
