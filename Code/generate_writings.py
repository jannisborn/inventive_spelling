# Task: Come up with alternative spellings for all orthographic words in a database
# Tool 1: IPA -> Graphem Konverter Wiki
# Tool 2: SAMPA -> IPA Konverter Wiki

# Idea: Take the phonetic sequence (SAMPA) of every word, convert it to an IPA sequence and
        # then convert that to all grapheme sequences
    
# Needs: A dictionary for SAMPA -> IPA (ideally non-ambiguous). A dictionary for IPA -> Text (ambigu.)

import numpy as np
import itertools
import os


path = 'data/gpl.cd'

with open(path, 'r') as file:

    raw_data = file.read().splitlines()
    words = []
    phons = []

    for ind,raw_line in enumerate(raw_data):

        line = raw_line.split("\\")

        
        if line[-2]: # Use only words that HAVE a SAMPA transcript (reduces from 51k to 37345)

            # exclude foreign words that have the 'æ' tone (SAMPA '{' ) like in
            # exclude foreign words that have the 'ɑ' tone (SAMPA 'A' )
            if not 'A' in line[-2] and not '{' in line[-2] and not '~' in line[-2]: 
                if not ('tS' in line[-2] and not 'tsch' in line[1]):

                    words.append(line[1].lower()) # Make spellings lowercase only
                    phons.append(line[-2]) # Using SAMPA notation

print("Size of dataset is", len(words), "samples")


## Helper Method:

def str_to_num_dataset(X,Y):
    """
    This method receives 2 lists of strings (input X and output Y) and converts it to padded, numerical arrays.
    It returns the numerical dataset as well as the dictionaries to retrieve the strings.
    """

    # 1. Define dictionaries 
    # Dictionary assignining a unique integer to each input character
    try:
        u_characters = set(' '.join(X)) 
    except TypeError:
        # Exception for TIMIT dataset (one phoneme is repr. by seq. of chars)
        print("TypeError occurred.")
        u_characters = set([quant for seq in X for quant in seq])

    char2numX = dict(zip(u_characters, range(len(u_characters))))

    # Dictionary assignining a unique integer to each phoneme
    try:
        v_characters = set(' '.join(Y)) 
    except TypeError:
        print("TypeError occurred.")
        v_characters = set([quant for seq in Y for quant in seq])
    char2numY = dict(zip(v_characters, range(1,len(v_characters)+1))) # Using 0 causes trouble for tf.edit_distance
    
    # 2. Padding
    # Pad inputs
    char2numX['<GO>'] = len(char2numX) 
    char2numX['<PAD>']  = len(char2numX) 
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

((phons_num, words_num), (phon_dict, word_dict)) = str_to_num_dataset(phons,words)
#print(len(word_dict),word_dict)
#print(len(phon_dict),phon_dict)
#print(words_num.shape, phons_num.shape)
#print(words_num[321,:], phons_num[321,:])

np.savez("data/celex.npz", words=words_num, phons=phons_num, word_dict=word_dict, phon_dict=phon_dict)

data = np.load('data/celex.npz')
sampa_dict = {key:data['phon_dict'].item().get(key) for key in data['phon_dict'].item()}
sampa_keys = list(sampa_dict.keys()) # Has 43 keys originally, 40 after excluding {, ~, A
print("Length of phonetic dict is ", len(sampa_dict), " and the keys are: \n", sampa_keys) # Has 43 keys


# Step 1: Make a sampa_ipa dict. How to: Go to SAMPA -> IPA tabelle, for every SAMPA char, check whether it 
# is in the CELEX korpus. If yes, look up example word from wiki in korpus and check whether it is the right sampa
# sign. If yes, look up on wiktionary example word in IPA and check whether output sign is correct.

sampa_ipa = dict()

# Vowels
sampa_ipa['i'] = 'i'
sampa_ipa[':'] = 'ː'
sampa_ipa['I'] = 'ɪ'
sampa_ipa['e'] = 'e'
sampa_ipa['E'] = 'ɛ'
sampa_ipa['y'] = 'y'   # meaning a real ü like in kühl
sampa_ipa['@'] = 'ə'
sampa_ipa['a'] = 'a'
sampa_ipa['u'] = 'u'
sampa_ipa['U'] = 'ʊ'
sampa_ipa['o'] = 'o'
sampa_ipa['O'] = 'ɔ'


# consonants
sampa_ipa['p'] = 'p'
sampa_ipa['b'] = 'b'
sampa_ipa['t'] = 't'
sampa_ipa['d'] = 'd'
sampa_ipa['k'] = 'k'
sampa_ipa['g'] = 'g'
sampa_ipa['f'] = 'f'
sampa_ipa['v'] = 'v'
sampa_ipa['s'] = 's'
sampa_ipa['z'] = 'z'
sampa_ipa['S'] = 'ʃ'
sampa_ipa['x'] = 'x'
sampa_ipa['h'] = 'h'
sampa_ipa['m'] = 'm'
sampa_ipa['n'] = 'n'
sampa_ipa['N'] = 'ŋ'
sampa_ipa['l'] = 'l'
sampa_ipa['r'] = 'r'
sampa_ipa['j'] = 'j'
sampa_ipa['Z'] = 'ʒ'
sampa_ipa['+'] = ''  # meaning a bit unclear
sampa_ipa['#'] = 'ˈ' # following syllabus carries primary intonation
sampa_ipa['|'] = 'ø' # meaning a bit unclear
sampa_ipa['/'] = 'œ' # usually SAMPA uses 9 instead of / for this 
sampa_ipa['Y'] = 'ʏ' # meaning more a 'oü' like in Müll

# These are 37 keys only, so 6 are missing. Remaining ones are:

# <GO>          not needed for alt. writing creation
# <PAD>         not needed for alt. writing creation
#    (SPACE)    not needed
# {             excluded some foreign words
# A             excluded some foreign words
# ~             excluded some foreign words


ipa = []
for samp in phons:
    s = []
    for char in samp:
        s.append(sampa_ipa[char])
    ipa.append(''.join(s))
print("Amount of IPA samples is", len(ipa), ". Some samples are: \n")
print(" WORD         ===>        SAMPA        ===>        IPA")
for k in range(25100, 25110):
    print(words[k]," => ", phons[k]," => ", ipa[k])


print(sampa_ipa.values())





ipa_graph = dict()
ipa_graph['t'] = ['t', 'd', 'tt', 'th', 'dt']
ipa_graph['ə'] = ['e']
ipa_graph['n'] = ['n', 'nn']
ipa_graph['s'] = ['s', 'ss', 'ß', 'c', 'z'] # excluded t for Patience, ce for Renaissance/Farce, zz for Jazz
ipa_graph['a'] = ['a', 'ah']
ipa_graph['r'] = ['r', 'rh', 'rr'] # excluded rrh for Zirrhose/Myrrhe
ipa_graph['l'] = ['l', 'll']
ipa_graph['ɛ'] = ['e', 'ae']
ipa_graph['f'] = ['f', 'v', 'ff', 'ph']
ipa_graph['g'] = ['g', 'gg', 'gh']
ipa_graph['ɪ'] = ['i']
ipa_graph['k'] = ['k', 'g', 'ck', 'c', 'ch', 'kk' ] # Excluded cch for Zucchini, gg for Flaggschiff, qu for Boutique
                  # Even kk is reasonable (Mokka, Akkordeon). qu chars are usually kv ipa (tracked below)
ipa_graph['m'] = ['m', 'mm']
ipa_graph['b'] = ['b', 'bb']
ipa_graph['ʃ'] = ['sch', 's'] # excluded sk for Ski, sh for Sheriff, Show and ch for Recherche 
ipa_graph['d'] = ['d','dd']
ipa_graph['p'] = ['p', 'b', 'pp'] # excluded bb for abebben or schrubben
ipa_graph['ŋ'] = ['ng','n']
ipa_graph['ɔ'] = ['o'] # excluded ch for Chauffeur (very rare exception)
ipa_graph['v'] = ['w', 'v']
ipa_graph['ʊ'] = ['u']
ipa_graph['z'] = ['s'] # excluded zz for Blizzard, Puzzle and z for zoomen, bulldozer (since only in foreign words)
ipa_graph['h'] = ['h']
ipa_graph['i'] = ['i', 'y'] # excluding y (Baby/Party/Hockey) would be reasonbale (only 10 words in corpus...)
ipa_graph['ʏ'] = ['ue', 'y', 'u'] # The corpus is weird here and writes Druck as drʏk, i.e. "Drück" rather than drʊk
ipa_graph['x'] = ['ch']
ipa_graph['e'] = ['e'] # excluded ee for Kaffee since IPA would be eː
ipa_graph['j'] = ['j', 'y']
ipa_graph['u'] = ['u'] # excluded ou like in Boutique
ipa_graph['o'] = ['o'] # not needed anyways since o always followed by ː
ipa_graph['œ'] = ['oe']
ipa_graph['y'] = ['y']
ipa_graph['ʒ'] = ['g', 'j'] # no wiki entry, self generated. For Garage or Jury


# 2 character keys:
ipa_graph['ts'] = ['z', 'ts', 'tts', 'tz', 't'] # excluded zz for Pizza/Skizze, c for circa, Penicillin
            # but t is reasonable for Aktion, Negation, Infektion, Proportion, ...
ipa_graph['aː'] = ['a', 'ah', 'aa']
ipa_graph['ai'] = ['ei', 'ai'] # excluded ail for Detail, aill for Medaillon, aille for Medaille and y for Nylon
ipa_graph['iː'] = ['ie', 'i', 'ieh', 'ih']
ipa_graph['eː'] = ['e', 'ee', 'eh'] # excluded et like in Bidet
ipa_graph['ɛː'] = ['ae', 'aeh']
ipa_graph['uː'] = ['u', 'uh'] # excluded ou like in Ragout, Limousine and oo like on zoomen/Cartoon
ipa_graph['oː'] = ['o', 'oh', 'oo'] # excluded au for aubergine/sauce and eau for plateau, Niveau
ipa_graph['yː'] = ['ue', 'ueh', 'y'] # excluded uet like in Debüt and u like in deja-vu
ipa_graph['ɔy'] = ['eu', 'aeu', 'oi', 'oy'] # instead of what wiki calls ɔɪ̯
ipa_graph['ks'] = ['chs', 'x', 'ks', 'gs'] #excluded gs like in legst/bugsieren and ggs like in eggst (?)
ipa_graph['øː'] = ['oe', 'oeh'] # excluded eu like in Ingenieur and eue like in Queue (?)
ipa_graph['kv'] = ['qu']


print(len(sampa_ipa), len(ipa_graph))
# We had 36 keys in sampa_ipa dict, now we have 46 already in ipa_graph due to 2-phoneme-groups
# But still there are one-char-values in sampa_ipa which are not keys in ipa_graph. Let us print them:
for key in sampa_ipa.values():
    if key not in ipa_graph.keys():
        print(key)
        
# Okay 4 are missing:
ipa_graph['ː'] = [''] # not needed anyways since ː always occurs after vowel
ipa_graph['ˈ'] = [''] # Just a pronounciation symbol, does not carry meaning for spelling
# Then the empty string '' is not needed as key
# Then ø only occurs followed by a ː





def split_word(word, ipa_graph):
    """
    Splits up an IPA word into a list of lists each with the possible replacement grapheme for each phoneme
    
    Parameters:
    ----------
    WORD       {list} in IPA notation
    IPA_GRAPH  {dict} mapping IPA symbols to possible grapheme sequences
    
    Returns:
    ---------
    CHARS      {list} containing lists with possible grapheme sequences
    """
    
    
    chars = []
    
    for ind in range(len(word)-1):
        
        if word[ind:ind+2] in ipa_graph:
            chars.append(ipa_graph[word[ind:ind+2]])
        else:
            chars.append(ipa_graph[word[ind]])
            
    chars.append(ipa_graph[word[ind+1]])
    
    return chars
    
all_writings = []
m = 0

for ind,ip in enumerate(ipa):
    if ind % 200 == 0:
        print("Currently examining word ", ind)
        
    word_lists = split_word(ip, ipa_graph)
    alt_write_raw = list(itertools.product(*word_lists))
    alt_write = [''.join(a) for a in alt_write_raw]
    try:
        alt_write.remove(words[ind])
    except ValueError:
        _ = 1
        
    all_writings.append(alt_write)
    
    if len(alt_write) > m:
        print(m,ind)
        m = len(alt_write)
        
        
print("DONE! Alternative writings generated for ", ind, "words. Resulting list has", len(all_writings), 'entries.')
np.save('celex_all_writings', all_writings)

