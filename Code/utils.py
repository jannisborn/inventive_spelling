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

        while start + BATCH_SIZE <= len(x):
            yield x[start:start+BATCH_SIZE], y[start:start+BATCH_SIZE], alt_targs[start:start+BATCH_SIZE]
            start += BATCH_SIZE

        
        
def accuracy(logits, labels, char2numY, mode='train'):
    """ 
    Receives the LOGITS and the LABELS (np.array of identical dimensions) 
    Return token accuracy (Levensthein distance), word accuracy and the initial measure of Sachin

    MODE can be either train or test. When train the actual logits are received, when test then "logit" holds already the prediction
    """

    # numpy: "a.argmax(-1)" reduces last dimension - identical to tensorflowL tf.argmax(a,axis=-1)

    # Error handling and mode setting
    if mode =='train':
        fullPred = logits.argmax(-1) # Prediction string with padding
    elif mode == 'test':
        fullPred = np.copy(logits.astype(int))
    else:
        print("Please specify 'mode' as either 'train' or 'test'. ")

    #print(logits.shape, labels.shape)
    #print(logits.dtype, labels.dtype)
    
    #Padded target string
    fullTarg = np.copy(labels) 
    # Set pads to 0 - as preparation for edit_distance
    if '<PAD>' in char2numY:
        fullPred[fullPred==char2numY['<PAD>']] = 0
        fullTarg[fullTarg==char2numY['<PAD>']] = 0
            
    # Initial measure in file
    oldAcc = np.mean(fullPred == fullTarg)     

    with tf.Session() as sess2:
    
        # Compute accuracy based on Levensthein Distance (without Padding) 
        dists = tf.edit_distance(dense_to_sparse(fullPred), dense_to_sparse(fullTarg)).eval(session=sess2)

        tokenAcc = 1 - tf.reduce_mean(dists).eval(session=sess2)
    wordAcc = np.count_nonzero(dists==0) / len(dists)
    return oldAcc, tokenAcc, wordAcc
            

def dense_to_sparse(denseTensor):
    ''' 
    Converts a dense tensor into a sparse one. Opposite to built-in tf.dense_tensor_to_sparse()
    '''

    # Create constant tensor 
    denseConst = tf.constant(denseTensor)

    # Non-Zero indices of dense tensor
    idx = tf.where(tf.not_equal(denseConst,0))

    # Create sparse tensor
    sparseTensor = tf.SparseTensor(idx, tf.gather_nd(denseConst,idx), denseConst.get_shape())
    '''with tf.Session() as sess:
        sparse = sess.run(sparseTensor)
    return sparse'''
    return sparseTensor





def date_dataset(seed):
    """
    This method retrieves and initializes a toy dataset used by Sachin Abeywardana (http://www.deepschool.io)
    This dataset can be used to verify code functionality
    """

    from faker import Faker
    import babel, random
    from babel.dates import format_date

    def create_date():
        """
            Creates some fake dates 
            :returns: tuple containing 
                      1. human formatted string
                      2. machine formatted string
                      3. date object.
        """
        dt = fake.date_object()
        try:
            human = format_date(dt, format=random.choice(FORMATS), locale=random.choice(LOCALES))
            case_change = random.randint(0,3) # 1/2 chance of case change
            if case_change == 1:
                human = human.upper()
            elif case_change == 2:
                human = human.lower()
            machine = dt.isoformat()
        except AttributeError as e:
            return None, None, None
        return human, machine #, dt


    fake = Faker()
    fake.seed(seed)
    random.seed(seed)

    FORMATS = ['short', 'medium', 'long', 'full', 'd MMM YYY', 'd MMMM YYY', 'dd MMM YYY', 'd MMM, YYY', 'd MMMM, YYY', 
                'dd, MMM YYY','d MM YY', 'd MMMM YYY', 'MMMM d YYY', 'MMMM d, YYY', 'dd.MM.YY']

    LOCALES = babel.localedata.locale_identifiers()
    LOCALES = [lang for lang in LOCALES if 'en' in str(lang)]

    data = [create_date() for _ in range(50000)]

    DX = [x for x, y in data] # A list of input strings
    DY = [y for x, y in data] # A list of output strings

    # Make dictionaries for character/numbers
    u_characters = set(' '.join(DX)) # All character in inputs
    Dchar2numX = dict(zip(u_characters, range(len(u_characters))))
    v_characters = set(' '.join(DY)) # All character in labels
    Dchar2numY = dict(zip(v_characters, range(len(v_characters))))

    # Padd inputs
    Dchar2numX['<PAD>'] = len(Dchar2numX)
    max_len = max([len(date) for date in DX])
    Dx = [[Dchar2numX['<PAD>']] * (max_len - len(date)) + [Dchar2numX[x_] for x_ in date] for date in DX]
    Dx = np.array(Dx)

    # Padd outputs
    Dchar2numY['<GO>'] = len(Dchar2numY)
    Dnum2charY = dict(zip(Dchar2numY.values(), Dchar2numY.keys()))
    Dy = [[Dchar2numY['<GO>']] + [Dchar2numY[y_] for y_ in date] for date in DY]
    Dy = np.array(Dy)

    return ((Dx,Dy),(Dchar2numX,Dchar2numY))





def str_to_num_dataset(X,Y,pads=5):
    """
    This method receives 2 lists of strings (input X and output Y) and converts it to padded, numerical arrays.
    It returns the numerical dataset as well as the dictionaries to retrieve the strings.
    
    PADS    {int} specifiying how many additional fields should be padded (to allow long words to have longer alt. writings)
    """

    # 1. Define dictionaries 
    # Dictionary assignining a unique integer to each input character
    try:
        u_characters = set(' '.join(X)) 
    except TypeError:
        # Exception for TIMIT dataset (one phoneme is repr. by seq. of chars)
        print("TypeError occurred.")
        u_characters = set([quant for seq in X for quant in seq])

    char2numX = dict(zip(u_characters, range(1,len(u_characters)+1)))

    # Dictionary assignining a unique integer to each phoneme
    try:
        v_characters = set(' '.join(Y)) 
    except TypeError:
        print("TypeError occurred.")
        v_characters = set([quant for seq in Y for quant in seq])
    char2numY = dict(zip(v_characters, range(1,len(v_characters)+1))) # Using 0 causes trouble for tf.edit_distance
    
    # 2. Padding
    # Pad inputs
    char2numX['<GO>'] = len(char2numX) + 1
    char2numX['<PAD>'] = len(char2numX) + 1
    mx_l_X = max([len(word) for word in X]) # longest input sequence
    # Padd all X for the final form for the LSTM
    x = [[char2numX['<GO>']] + [char2numX['<PAD>']]*(mx_l_X - len(word)) +[char2numX[char] for char in word] for word in X]
    x = np.array(x) 

    # Pad targets
    char2numY['<GO>'] = len(char2numY) + 1 # Define number denoting the response onset
    char2numY['<PAD>'] = len(char2numY) + 1 
    mx_l_Y = max([len(phon_seq) for phon_seq in Y]) # longest output sequence

    y = [[char2numY['<GO>']] + pads*[char2numY['<PAD>']] + [char2numY['<PAD>']]*(mx_l_Y - len(ph_sq)) + [char2numY[phon] for phon in ph_sq] for ph_sq in Y]
    y = np.array(y)

    return ((x,y) , (char2numX,char2numY))



def TIMIT_G2P():
    """
    This method uses NLTK to prepare a sample from the TIMIT corpus (https://catalog.ldc.upenn.edu/ldc93s1) for a Grapheme-To-Phoneme conversion.
    The sample contains 660 words and their corresponding phoneme sequences (in their special phono-code) 

    """

    import nltk
    timitdict = nltk.corpus.timit.transcription_dict()

    # 1. Split data into inputs and targets
    X = [word for word in timitdict] # X is a list with 660 strings
    Y = [timitdict[word] for word in timitdict] # Y is a list with 660 lists, each containing of a few strings (one per phoneme)

    return str_to_num_dataset(X,Y)
    # All characters in inputs (28, i.e. alphabet + space and ' ) -> Final sequence length = 29 (due to padding)
    # All target phonemes (74 different) -> 76 (Pad, Go)
    

    """
    # 2. Define dictionaries 
    # Dictionary assignining a unique integer to each input character
    u_characters = set(' '.join(X)) # All characters in inputs (28, i.e. alphabet + space and ' ) -> Final sequence length = 29 (due to padding)
    char2numX = dict(zip(u_characters, range(len(u_characters))))
    # Dictionary assignining a unique integer to each phoneme
    v_characters = set([phon for phon_seq in Y for phon in phon_seq]) # All target phonemes (74 different) -> 76 (Pad, Go)
    char2numY = dict(zip(v_characters, range(1,len(v_characters)+1))) # Using 0 causes trouble for tf.edit_distance

    # 3. Padding
    # Pad inputs
    char2numX['<PAD>'] = len(char2numX) 
    mx_l_X = max([len(word) for word in X]) # longest input sequence
    # Padd all X for the final form for the LSTM
    x = [[char2numX['<PAD>']]*(mx_l_X - len(word)) +[char2numX[char] for char in word] for word in X]
    x = np.array(x) 

    # Pad targets
    char2numY['<GO>'] = len(char2numY) # Define number denoting the response onset
    char2numY['<PAD>'] = len(char2numY)  
    mx_l_Y = max([len(phon_seq) for phon_seq in Y]) # longest output sequence

    y = [[char2numY['<GO>']] + [char2numY['<PAD>']]*(mx_l_Y - len(ph_sq)) + [char2numY[phon] for phon in ph_sq] for ph_sq in Y]
    y = np.array(y)

    return ((x,y) , (char2numX,char2numY))
    """



def TIMIT_P2G():
    """ 
    This method uses NLTK to prepare a sample from the TIMIT corpus (https://catalog.ldc.upenn.edu/ldc93s1) for a Phoneme-To-Grapheme conversion.
    The sample contains 660 words and their corresponding phoneme sequences (in their special phono-code)  
    """

    import nltk
    timitdict = nltk.corpus.timit.transcription_dict()

    # 1. Split data into inputs and targets
    X = [timitdict[word] for word in timitdict] # inputs are phonemes
    Y = [word for word in timitdict] # outputs are words

    return str_to_num_dataset(X,Y)


    """
    # 2. Define dictionaries 
    # Dictionary assignining a unique integer to each phoneme
    v_characters = set([phon for phon_seq in X for phon in phon_seq]) 
    char2numX = dict(zip(v_characters, range(len(v_characters)))) 
    # Dictionary assignining a unique integer to each input character
    u_characters = set(' '.join(Y))
    char2numY = dict(zip(u_characters, range(1,len(u_characters)+1)))

    # 3. Padding
    char2numX['<PAD>'] = len(char2numX)  # Define number denoting a padded output
    mx_l_X = max([len(phon_seq) for phon_seq in X]) # longest input sequence
    x = [[char2numX['<PAD>']]*(mx_l_X - len(ph_sq)) + [char2numX[phon] for phon in ph_sq] for ph_sq in X]
    x = np.array(x)

    char2numY['<GO>'] = len(char2numY) # Define number denoting the response onset
    char2numY['<PAD>'] = len(char2numY) # Define number denoting a padded input
    mx_l_Y = max([len(word) for word in Y]) # longest input sequence
    y = [[char2numY['<GO>']] + [char2numY['<PAD>']]*(mx_l_Y - len(word)) +[char2numY[char] for char in word] for word in Y]
    y = np.array(y) 

    return ((x,y) , (char2numX,char2numY))
    """

def np_dict_to_dict(np_dict):
    """
    Converts a dictionary saved via np.save (as structured np array) into an object of type dict

    Parameters:
    --------------
    NP_DICT        : {np.array} structured np.array with dict keys and items

    Returns:
    --------------
    DICT            : {dict} converted NP_DICT

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



def BAS_json(path):
    """
    This method receives a path for the BAS-SprecherInnen corpus and iterates through all JSON files in all subfolders.
    It creates and returns a list of words and a list of pronounciations
    """
    
    import json, os

    words = []
    prons = []
    ind = 0
    # Read in filenames
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in [f for f in filenames if f.endswith(".json")]:

            if filename == 'SprecherInnen_DBconfig.json':
                continue

            # Open the json
            with open(os.path.join(dirpath,filename)) as json_file:
                data = json.load(json_file)

                for item in data['levels'][1]['items']:
                    words.append(item['labels'][0]['value'])
                    prons.append(item['labels'][1]['value'])

    return words,prons


def clean_corpus_BAS_Sprecherinnen(words,prons):
    """
    This method receives a list of words and a list of pronunciations of the BAS-Sprech. corpus and returns a cleaned dataset.
    Clearning means:    1) Removing multiple occurrences of words       2) Remove misspellings and ambiguities
                        3) Remove capitalization at begin of sentence   
    Homophon words (Meer, mehr) are kept!

    This method required manual inspection (once for each corpus).
    Returns a condensed list of words and pronounciations (strings) that can be converted in numerical values next.

    """

    # First, we remove multiple occurrences.
    # We cannot use set(words), set(prons) since some words are homophon 8(results in diff. lengths)
    all_tups = []
    for (w,p) in zip(words,prons):
        all_tups.append(tuple((w,p)))
    set_tup = set(all_tups)
    print('Amount of non-unique words in corpus is ', len(all_tups))
    unique_tups = dict(set_tup)
    print('Amount of unique words in corpus is ', len(set_tup))

    # Now we have removed multiple occurrences and we have a dict of tuples (word, pron)

    def find_poss_mistakes(unique_tups):
        """
        Receives a list of hopefully unique tupels (word,pron) and collect the tupels
        which may have incorrect spelling/pronounciations.
        """
        possible_mistakes = []
        for key, val in unique_tups.items():
            for keyy,vall in unique_tups.items():
                if key != keyy and val == vall:
                    # Detect multiple spellings of same pronounciation
                    possible_mistakes.append((key,val, keyy, vall))
                if key == keyy and val != vall:
                    # Detect multiple pronounciations of same spelling
                    possible_mistakes.append((key,val, keyy, vall))
                    
        return possible_mistakes
        
    poss_mist = find_poss_mistakes(unique_tups)
    """
    print("+++ Possible mistakes are +++")
    for k in range(len(poss_mist)):
        print(poss_mist[k][0],' -> ',poss_mist[k][1], 
              poss_mist[k][2],' -> ',poss_mist[k][2])
    """
        
    # Remove mistakes (after manual inspection)
    unique_tups.pop('BäckerInnen') # removing as a duplicate of Bäckerinnen
    unique_tups.pop('nu') # Duplicate of Nu
    unique_tups.pop('Abonentinnen') # Misspelled
    unique_tups.pop('Mit') # Duplicate of mit
    unique_tups.pop('A') # Duplicate of ah
    unique_tups.pop('Bei') # Duplicate of bei
    unique_tups.pop('backwaren') # Duplicate of Backwaren
    unique_tups.pop('-vertreterinnen') # Duplicate of Vertreterinnen
    unique_tups.pop('leu') # Duplicate of Leu
    unique_tups.pop('teil') # Duplicate of Teilt
    unique_tups.pop('Un') # Duplicate of un
    unique_tups.pop('Ver') # Duplicate of ver
    unique_tups.pop('AutorInnen') # Duplicate of Autorinnen
    unique_tups.pop('FreundInnen') # Duplicate of Freundinnen
    unique_tups.pop('-pflegerin') # Duplicate of Pflegerin
    unique_tups.pop('Neu') # Duplicate of neu
    unique_tups.pop('re') # Duplicate of Re
    unique_tups.pop('-kolleginnen') # Duplicate of Koleginnen
    unique_tups.pop('-trinkerinnen') # Duplicate of Trinkerinnen
    unique_tups.pop('Twitter-NutzerInnen') # Duplicate of Twitter-Nutzerinnen
    unique_tups.pop('kommissionen') # Duplicate of Koleginnen
    # Remaining: (Ihnen, ihnen), (dass,das), (Meer, mehr), (Ihres, ihres), (mal, Mal)

    wordss = list(unique_tups.keys())
    pronss = list(unique_tups.values())
    print('After clearning ', len(wordss), ' different words remain')

    return wordss, pronss




def BAS_P2G_create(path):
    """
    Phoneme-To-Grapheme Conversion (Writing)
    This method receives the path to the BAS-Sprecherinnen corpus and creates the numerical dataset.
    It returns inputs and labels as tupel and their corresponding char2num dictionaries
    """

    words, prons = BAS_json(path)
    words, prons = clean_corpus_BAS_Sprecherinnen(words,prons)
    return str_to_num_dataset(prons, words)


def BAS_G2P_create(path):
    """
    Grapheme-To-Phoneme Conversion (Reading) 
    This method receives the path to the BAS-Sprecherinnen corpus and creates the numerical dataset.
    It returns inputs and labels as tupel and their corresponding char2num dictionaries
    """
    words, prons = BAS_json(path)
    words, prons = clean_corpus_BAS_Sprecherinnen(words,prons)
    return str_to_num_dataset(words,prons)



def BAS_P2G_retrieve():
    """
    Shortcut method for quickly retrieving numerical dataset of BAS-Sprecher corpus
    In case whole dataset is not copied on remote machine
    """
    data = np.load('data/BAS.npz')
    input_dict = np_dict_to_dict(data['inp_dict'])
    target_dict = np_dict_to_dict(data['tar_dict'])

    return ( (data['inputs'], data['targets']) , (input_dict, target_dict) )

def BAS_G2P_retrieve():
    """
    Shortcut method for quickly retrieving numerical dataset of BAS-Sprecher corpus
    In case whole dataset is not copied on remote machine
    """
    data = np.load('data/BAS_G2P.npz')
    input_dict = np_dict_to_dict(data['inp_dict'])
    target_dict = np_dict_to_dict(data['tar_dict'])
    return ( (data['inputs'], data['targets']) , (input_dict, target_dict) )



def write_word(model, phon_word, phon_dict, orth_dict):
    """
    Receives a phonetic sequence and writes the word down. Prints input and output

    Parameters:
    --------------
    MODEL           {bLSTM} the trained bLSTM object 
    PHON_WORD       {str, np.array} either a plain string of phonemes or a np.array
    PHON_DICT       {dict} mapping the phonemes to the integers
    ORTH_DICT       {dict} mapping the characters to integers        

    Returns:
    ------------------
    No returns

    """

    # Error 
    if isinstance(phon_word,np.ndarray) and phon_word.dtype==int:
        phon_word_num = phon_word
        rev_dict = dict(zip(phon_dict.values(), phon_dict.keys()))
        phon_word = [rev_dict[phon] for phon in phon_word_num]
    elif isinstance(phon_word,str):
        l = model.input_seq_length
        phon_word_num = [phon_dict[phon_word[-k]] if k<len(phon_word) else phon_dict['<PAD>'] for k in range(l,0,-1)]
        
        #phon_word_num = np.array([phon_dict[phon] for phon in phon_word])
        phon_word_num = np.expand_dims(phon_word_num, axis=0)
    else:
        raise TypeError("Please insert 2nd argument (phon_word) either as a string or as a np.array of type int")

    if not model.task == 'write':
        raise ValueError("Please insert as 1st argument a model (type bLSTM) that was trained on a writing task.")
    orth_dict_rev = dict(zip(orth_dict.values(), orth_dict.keys()))


    with tf.Session() as sess:
        
        sess.run(tf.global_variables_initializer())

        dec_input = np.zeros([1,1]) + phon_dict['<GO>']

        for k in range(phon_word_num.shape[1]):
            logits = sess.run(model.logits, feed_dict={model.keep_prob:1.0, model.inputs:phon_word_num, model.outputs:dec_input})
            char = logits[:,-1].argmax(axis=-1)
            dec_input = np.hstack([dec_input, char[:,None]])
        written = ''.join([orth_dict_rev[num] for num in dec_input[0,:]])
        
        print("The spoken sequence ", phon_word, "  =>  ", written)
        
        




def read_word(model, orth_word, orth_dict, phon_dict):
    """
    Opposite to write_word, receives an orthographic sequence and reads it oud loud
    
    Parameters:
    --------------
    MODEL           {bLSTM} the trained bLSTM object 
    ORTH_WORD       {str, np.array} either a plain string of phonemes or a n
    ORTH_DICT       {dict} mapping the letters to the integers        
    PHON_DICT       {dict} mapping the phonemes to the integers        

    Returns:
    ------------------
    No returns

    """

    # Error 
    if isinstance(orth_word,np.ndarray) and orth_word.dtype==int:
        orth_word_num = orth_word
        rev_dict = dict(zip(orth_dict.values(), orth_dict.keys()))
        orth_word = [rev_dict[orth] for orth in orth_word_num]
    elif isinstance(orth_word,str):
        orth_word_num = [orth_dict[orth] for orth in orth_word]
    else:
        raise TypeError("Please insert 2nd argument (orth_word) either as a string or as a np.array of type int.")

    if not model.task == 'read':
        raise ValueError("Please insert as 1st argument a model (type bLSTM) that was trained on a reading task.")

    phon_dict_rev = dict(zip(phon_dict.values(), phon_dict.keys()))

    with tf.Session() as sess:

        dec_input = np.zeros([1,1]) + orth_dict['<GO>']
        for k in range(orth_word):
            logits = sess.run(model.logits, feed_dict={model.keep_prob:1.0, model.inputs:orth_word, model.outputs:dec_input})
            char = logits[:,-1].argmax(axis=-1)
            dec_input = np.hstack([dec_input, char[:,None]])

        spoken = [phon_dict_rev[num] for num in dec_input]

        print("The written sequence ", orth_word, "  =>  ", dec_input)


def retrieve_model(path, num):
    """
    Retrieves a trained bLSTM model from the specified path

    Parameters:
    -----------------
    PATH        {str} the path to the folder in which the model lies
    NUM         {int} the ID of the model (#epochs)

    """
    with tf.Session() as sess:
        pretrained_model = tf.train.import_meta_graph(path+'/Model-'+str(num)+'.meta')
        pretrained_model.restore(sess,tf.train.latest_checkpoint(path)) 

    print(type(pretrained_model))

    return saver


path = '/Users/jannisborn/Desktop/LDS_Data/celex2/german/gpl/gpl.cd'

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
    path = "/Users/jannisborn/Desktop/LDS_Data/celex2/german/gpl/gpl.cd"
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



def celex_retrieve(learn_type):
    """
    Retrives the previously saved data from the CELEX corpus
    """

    data = np.load('../../Models/celex.npz')
    #data = np.load('../../Models/data/celex_small.npz')
    phon_dict = np_dict_to_dict(data['phon_dict'])
    word_dict = np_dict_to_dict(data['word_dict'])
    

    if learn_type == 'lds':

        data = np.load('../../Models/data/celex_small.npz')
        phon_dict = np_dict_to_dict(data['phon_dict'])
        word_dict = np_dict_to_dict(data['word_dict'])
        path = '../../Models/data/celex_alt_targets_small.npy'
        print("Loading alternative targets ...")
        alt_targs_raw = np.load(path)

        ## Workaround due to pickle bug (cannot save npy files > 2GB)
        """
        max_bytes = 2**31 - 1
        input_size = os.path.getsize(filepath)
        bytes_in = bytearray(0)
        with open(filepath, 'rb') as f_in:
            for _ in range(0, input_size, max_bytes):
                bytes_in += f_in.read(max_bytes)
        alt_targs_raw = pickle.loads(bytes_in) # Takes a couple of minutes
        
        """

        ## Workaround to avoid returning a np object (array of lists) -> Yields memory error
        """
        max_len = max([len(l) for l in alt_targs_raw])
        inp_seq_len = len(alt_targs_raw[1][0])
        alt_targs = np.zeros([len(alt_targs_raw), inp_seq_len, max_len], dtype=np.int8)
        for word_ind in range(len(alt_targs_raw)):
            for write_ind in range(len(alt_targs_raw[word_ind])):
                alt_targs[word_ind,:,write_ind] = np.array(alt_targs_raw[word_ind][write_ind],dtype=np.int8)

        """

        alt_targs = np.array([np.array(d,dtype=np.int8) for d in alt_targs_raw])
        print("Alternative targets successfully loaded.")

        return ( (data['phons'], data['words']) , (phon_dict, word_dict), alt_targs )
    else:
        return ( (data['phons'], data['words']) , (phon_dict, word_dict))


def childlex_retrieve():
    """
    Retrives the previously saved data from the childlex database (subset of CELEX)
    """

    data = np.load('data/childlex.npz')
    phon_dict = np_dict_to_dict(data['phon_dict'])
    word_dict = np_dict_to_dict(data['word_dict'])

    return ( (data['phons'], data['words']) , (phon_dict, word_dict))

def fibel_retrieve():

    data = np.load('data/fibel.npz')
    phon_dict = np_dict_to_dict(data['phon_dict'])
    word_dict = np_dict_to_dict(data['word_dict'])

    return ( (data['phons'], data['words']) , (phon_dict, word_dict))



def get_last_id(dataset):

    # Retrieves the maximal ID of all the saved models in the path

    from os.path import expanduser
    
    root_local = expanduser("~")+'/Dropbox/GitHub/LSTM/Models/'
    path = root_local + dataset

    try:
        folders = next(os.walk(path))[1]
    except StopIteration:
        pass
    IDs = [int(s) for folder in folders for s in folder.split('_') if s.isdigit()]

    return max(IDs)


def lds_compare(prediction, targets, alt_targets):
    """
    Like in sequence_loss_lds this method checks whether the generated predictions match any of the alternative targets

    Parameters:
    --------------
    PREDICTION      {np.array}  2D  of shape batch_size x seq_len
    TARGETS         {np.array}  2D  of shape batch_size x seq_len
    ALT_TARGETS     {np.array}  3D  of shape batch_size x seq_len x max_alt_writings

    Returns:
    --------------
    NEW_TARGETS     {np.array}  2D  of shape batch_size x seq_len

    """
    batch_size = targets.shape[0]
    max_alt_spellings = alt_targets.shape[2]


    new_targets = np.zeros(targets.shape, np.int64)
    counter = []

    for wo_ind in range(batch_size):
        wrote_alternative = False

        # Check whether the word was actually correctly spelled.
        if np.array_equal(prediction[wo_ind,:],targets[wo_ind,:]):
            new_targets[wo_ind,:] = targets[wo_ind,:]


        else:
            # If not, check all the alternative writings
            for tar_ind in range(max_alt_spellings):
                if np.array_equal(prediction[wo_ind,:], alt_targets[wo_ind,:,tar_ind]):
                    wrote_alternative = True 
                    new_targets[wo_ind,:] = alt_targets[wo_ind,:,tar_ind]
                    continue

        # In case the spelling was actuall bullshit
        if not wrote_alternative:
            new_targets[wo_ind,:] = targets[wo_ind,:]

        counter.append(wrote_alternative)
    
    # How many words were written alternatively?
    rat = sum(counter) / len(counter)
    if rat > 0:
          print("Ratio of words that were 'correct' in LdS sense: ", str(rat))


    return new_targets





                                               





               
