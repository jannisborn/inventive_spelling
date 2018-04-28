import warnings
warnings.filterwarnings("ignore",category=FutureWarning)
import tensorflow as tf 
import numpy as np 
import os, sys, time, argparse
import utils
from bLSTM import bLSTM


"""
Script to evaluate performance of a model

"""





if __name__ == '__main__':

	parser = argparse.ArgumentParser()

	parser.add_argument('--dataset', default='celex', type=str,
						help='The dataset on which the model was trained, from {celex, bas, bas_p2g_r}')
	parser.add_argument('--learn_type', default='normal', type=str,
						help='The used learning paradigm. Choose from {normal, lds}.')
	parser.add_argument('--task', default='write', type=str,
						help="The task the model solved. Choose from {write, read}.")

	args = parser.parse_args()

	parser.add_argument('--id', default=utils.get_last_id(args.dataset), type=int,
						help='The ID of the model that should be examined. Per default the last in the folder')
	parser.add_argument('--epochs', default=None, type=int, 
						help="The timestamp (in epochs) of the model. Default=None (after last epoch).")

	args = parser.parse_args()








class evaluation(object):

	"""
	This class is to evaluate performance of an already trained model

	"""


	def __init__(self, args):

		# Transfer args to class
		self.dataset = args.dataset
		self.learn_type = args.learn_type
		self.task = args.task
		self.id = args.id
		self.epochs = args.epochs

		# Receives the path to the folder of a stored model
		self.root_local = expanduser("~")+'/Dropbox/GitHub/LSTM/'
		self.path = self.root_local + 'Models/' + self.dataset + '/' + self.learn_type + '_run_' + str(self.id) + '/'

		# Retrieve relevant data
		self.retrieve_dicts()
		self.retrieve_model_args()
		self.retrieve_model()



	def retrieve_model_args(self):
		"""
		Retrieves the hyperparameters of the trained bLSTM (#layers, #nodes, learning rate etc.) which is saved in meta_tags.csv
		"""
		import pandas as pd


		df = pd.read_csv(self.path+'/test_tube_data/version_0/meta_tags.csv')
		raw_args = df['value'].values.tolist()

		self.model_args = []
		types = ['i','i','i','i','i','i','i','i','i',
		         's','s','b','s','f','s','f','s',
		         'b','b','i','i','b','f','l',',l'] # ends with test_indices
		for ind,raw_arg in enumerate(raw_args):
		    if types[ind] == 'i':
		        self.model_args.append(int(raw_arg))
		    elif types[ind] == 's':
		        self.model_args.append(str(raw_arg))
		    elif types[ind] == 'b':
		        self.model_args.append(raw_arg==True)  
		    elif types[ind] == 'f':
		        self.model_args.append(float(raw_arg))
		    elif types[ind] == 'l':
		    	self.model_args.append(list(raw_arg))




	def retrieve_dicts(self):
		"""
		Retrieves the dictionary to map characters -> digits for the used dataset (task)
		"""

		path = self.root_local + 'LdS_bLSTM/Code/data/'
		data = np.load(path + self.task + '.npz')

		self.inputs = data['inputs']
		self.targets - data['targets']

		# Depending on whether the task is to read or to write, dictionaries need to be flipped.
		self.input_dict = {key:data['inp_dict'].item().get(key) for key in data['inp_dict'].item()} if self.learn_type == 'write' else {key:data['tar_dict'].item().get(key) for key in data['tar_dict'].item()}
		self.output_dict = {key:data['tar_dict'].item().get(key) for key in data['tar_dict'].item()} if self.learn_type == 'write' else  {key:data['inp_dict'].item().get(key) for key in data['inp_dict'].item()}


	def retrieve_model(self):
		"""
		Initializes a new instance of the model, with identical arguments to the trained one. 
		Later, weights will be restored.
		"""

		self.net = blSTM(*self.model_args[:13])
		self.net.forward()
		self.net.backward()





	def predict_input():
		"""
		Use this method for command line interaction with the model (showing its predictions to user-specified input words/phonemes).
		Leave method with pressing <SPACE>
		"""
		loop = True
		inp = 'phonetic' if args.task == 'write' else 'orthografic' # To read in a type of sequence
		out = 'spoken' if self.task == 'write' else 'written'


		out_dict_rev = dict(zip(self.output_dict.values(), self.output_dict.keys()))


		with tf.Session() as sess:

			# Restore model
			saver = tf.train.Saver(tf.global_variables())
			saver.restore(sess,tf.train.latest_checkpoint(self.path))

			while loop:

				word = input("Please insert a ", inp, " sequence")

				if word == ' ':
					loop = False
					break

				word_num = self.prepare_sequence(word)

				dec_input = np.zeros([1,1]) + self.input_dict['<GO>']

				for k in range(word_num.shape[1]):
					logits = sess.run(net.logits, feed_dict={net.keep_prob:1.0, net.inputs:phon_word_num, net.outputs:dec_input})
					char = logits[:,-1].argmax(axis=-1)
					dec_input = np.hstack(dec_input, char[:,None]) # Identical to np.expand_dims(char,1)

				dec_input = np.expand_dims(np.squeeze(dec_input)[np.squeeze(dec_input)!=0],axis=0)
				written = ''.join([orth_dict_rev[num] if orth_dict_rev[num]!='<PAD>' else '' for ind,num in enumerate(dec_input[0,1:])])
				print("The ", out, " sequence ", phon_word, "  =>  ", written)



	def prepare_sequence(self,word):
		"""
		Prepares a user-inserted word such that it can be feed into the model

		Parameters:
		--------------
		WORD 		{str}
		"""
		
		# Error handling
		if any(char.isdigit() for char in word):
			raise TypeError("Please insert a string that contains no numerical values.")

		l = self.model_args[0] # length of the input sequence 
		phon_word_num = [phon_dict[word[-k]] if k<=len(word) else self.input_dict['<PAD>'] for k in range(l,0,-1)]

		return np.expand_dims(phon_word_num, axis=0)



	def show_mistakes(mode='train'):
		"""
		Show the mistakes of the model on training or testing data and saves the mistakes to a .txt file

		Parameters:
		-----------
		MODE 	{str} either train or test
		"""

		self.indices = self.model_args[23] if mode=='train' else self.model_args[24] # Indices are either train or test indices
		out = 'spoken' if self.task == 'write' else 'written'

		with tf.Session() as sess:

			# Restore model
			saver = tf.train.Saver(tf.global_variables())
			saver.restore(sess,tf.train.latest_checkpoint(self.path))


			# Iterate over dataset and print all wrong predictions

			tested_inputs = self.inputs[self.indices]
			tested_labels = self.outputs[self.indices]
			dec_input = np.zeros((len(tested_words,1))) + self.input_dict['<GO>']

			# Classify
			for k in range(self.model_args[1]): # Length of output sequence

				logits = sess.run(net.logits, feed_dict={net.keep_prob:1.0, net.inputs:tested_inputs, net.outputs:dec_input})
				predictions = logits[:,-1].argmax(axis=-1)
				dec_input = np.hstack(dec_input, predictions[:,None])


			write_oldAcc, write_tokenAcc , write_wordAcc = utils.accuracy(sess,dec_input[:,1:], tested_labels[:,1:], self.output_dict , mode='test')
			print('Accuracy on {:6.3s} set is for tokens{:>6.3f} and for words {:>6.3f}'.format(mode,write_tokenAcc, write_wordAcc))

			print('\n',"Now printing the mistakes on the ", mode, " dataset")
			file = open('mistakes_'+mode+'_data.txt')
			for ind,pred in enumerate(dec_input[:,1:]):

				if pred != tested_labels[ind,1:]:

					inp_str = [self.inp_dict[k] if self.inp_dict[k] != '<PAD>' else '' for k in tested_inputs[ind,:]]
					out_str = [self.out_dict[k] if self.out_dict[k] != '<PAD>' else '' for k in pred]
					tar_str = [self.out_dict[k] if self.out_dict[k] != '<PAD>' else '' for k in tested_labels[ind,1:]]

					print("The ", out, " sequence ", inp_str , "  =>  ", out_str, ' instead of ', tar_str, file=file)
			file.close()



	def plot_inp_embedding():
		# tSNE, PCA etc.
		k =3


	def plot_out_embedding():

		k=3



