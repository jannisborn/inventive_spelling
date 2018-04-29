import warnings
warnings.filterwarnings("ignore",category=FutureWarning)
import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt 
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
		self.root_local = os.path.expanduser("~")+'/Dropbox/GitHub/LSTM/'
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


		output_dict_rev = dict(zip(self.output_dict.values(), self.output_dict.keys()))


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
			file = open('mistakes_'+mode+'_data.txt','w')
			for ind,pred in enumerate(dec_input[:,1:]):

				if pred != tested_labels[ind,1:]:

					inp_str = [self.input_dict[k] if self.input_dict[k] != '<PAD>' else '' for k in tested_inputs[ind,:]]
					out_str = [self.output_dict[k] if self.output_dict[k] != '<PAD>' else '' for k in pred]
					tar_str = [self.output_dict[k] if self.output_dict[k] != '<PAD>' else '' for k in tested_labels[ind,1:]]

					print("The ", out, " sequence ", inp_str , "  =>  ", out_str, ' instead of ', tar_str, file=file)
			file.close()



	def plot_pca(self, n_comp=2, mode='input', plot=True):
		"""
		PCA dimensionality reduction of the bLSTM's weight vectors. Plots weight vectors on first 2 eigenvectors.

		Parameters:
		-------------
		N_COMP 		{int} number of PCs to keep. Use 2 if plot=True, can be anything if called from plot_tsne for the sake of preprocessing
		MODE 		{str} choose from {'input','output'} depending on whether the input or output embedding vectors should be plotted
		PLOT 		{bool} to decide whether the plots should be displayed and saved (or the pca should just be computed, as preprocessing for tSNE)

		Returns:
		-------------
		PCS			{np.array} embedding vectors projected at the first 2 pcs. Shape: input_dict_size x 2

		"""
		from sklearn.decomposition import PCA
		from sklearn.preprocessing import StandardScaler


		with tf.Session() as sess:

			saver = tf.train.Saver(tf.global_variables())
			saver.restore(sess,tf.train.latest_checkpoint(self.path))

			if mode=='input':
				weight_vectors = self.net.input_embedding.eval()
				dic = dict(zip(self.input_dict.values(), self.input_dict.keys()))
				ling = 'phonetic' if args.task == 'write' else 'orthografic'
			elif mode == 'output':
				weight_vectors = self.net.output_embedding.eval()
				dic = dict(zip(self.output_dict.values(), self.output_dict.keys()))
				ling = 'orthografic' if args.task == 'write' else 'phonetic'
			else:
				raise ValueError("Specify mode as either 'input' or 'output'." )

			pca = PCA(n_components=n_comp)
			pcs = pca.fit_transform(weight_vectors)

			print("The explained variance of the first ", n_comp, ' PCs is (in %): ', np.round(np.sum(pca.explained_variance_ratio_),3))    
    

			if plot:


				fig = plt.figure(figsize = (8,8))
				ax = fig.add_subplot(1,1,1) 
				ax.set_xlabel('Principal Component 1', fontsize = 15)
				ax.set_ylabel('Principal Component 2', fontsize = 15)
				ax.set_title(['Projection of the '+ling+' vectors on the first 2 PCs.'], fontsize = 20)
				ax.scatter(pcs[:,0], pcs[:,1],s=2)
				ax.spines['right'].set_visible(False)
				ax.spines['top'].set_visible(False)
				for k in range(len(pcs)):
				    ax.annotate(dic[k],(pcs[k,0], pcs[k,1]))

				plt.savefig("PCA-Results.pdf")

				np.save("PCA_Results", pcs)

			else:

				return pcs



	def plot_tsne(self, perplexity=10, steps=5000, lr=10, init='random', angle=0.5, mode='input', pca=None):
		"""
		t-SNE dimensionality reduction of the bLSTM's weight vectors.

		Parameters:
		------------
		PERPLEXITY 	{int}, tunable hyperparameter, from [5,50] says author. Higher -> More attention to global aspects of data. 
						More samples -> Higher perplexity, perplexity should always < num_samples (default=30)
		STEPS 		{int}, tunable hyperparameter, amount of iterations, (default=100)
		LR 			{int}, tunable hyperparameter. If too high -> data is circular and equidistant(!) in embedded space. 
						If too low -> points compressed in clouds
		INIT 		{str}, choose from {'random', 'pca'}, the initialization of the embedding space (default=random)
		ANGLE 		{float}, from [0.0, 1.0], trade-off between accuracy (0.0) and speed (1.0) - default=0.5
		MODE 		{str} choose from {'input','output'} depending on whether the input or output embedding vectors should be plotted
		PCA 		{None,int} None per default, if integer is given, data is preprocessed with principal component analysis and 
						first PCA PCs are kept.		

		About t-SNE:
			1)		Repeated runs with the same data and hyperparameters give different results
			2)		Cluster sizes usually do not mean anything
			3) 		Distances between clusters may not mean anything

		"""

		from sklearn.manifold import TSNE 


		with tf.Session() as sess:

			saver = tf.train.Saver(tf.global_variables())
			saver.restore(sess,tf.train.latest_checkpoint(self.path))

			if mode=='input':
				weight_vectors = self.net.input_embedding.eval()
				dic = dict(zip(self.input_dict.values(), self.input_dict.keys()))
				ling = 'phonetic' if args.task == 'write' else 'orthografic'
			elif mode == 'output':
				weight_vectors = self.net.output_embedding.eval()
				dic = dict(zip(self.output_dict.values(), self.output_dict.keys()))
				ling = 'orthografic' if args.task == 'write' else 'phonetic'
			else:
				raise ValueError("Specify mode as either 'input' or 'output'." )


			if perplexity >= weight.vectors.shape[0]:
				raise ValueError("Please make sure the perplexity argument is smaller than the number of data points.")


			if pca is not None:
				# Preprocess via PCA
				weight_vectors = self.plot_pca(n_comp=pca, mode=mode, plot=False)

			t = time.time()
			tsne = TSNE(n_components=2, verbose=1, perplexity=perplexity, n_iter=steps, learning_rate=lr, init=init, angle=angle)
			tsne_results = tsne.fit_transform(weight_vectors)
			print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-t))


			fig = plt.figure(figsize = (8,8))
			ax = fig.add_subplot(1,1,1) 
			ax.set_xlabel('x-tsne', fontsize = 15)
			ax.set_ylabel('y-tsne', fontsize = 15)
			ax.set_title('tSNE '+ling+' vectors', fontsize = 20)

			ax.spines['right'].set_visible(False)
			ax.spines['top'].set_visible(False)

			ax.scatter(tsne_results[:,0], tsne_results[:,1],s=2)
			for k in range(len(pcs)):
			    ax.annotate(rev_in_dict[k],(tsne_results[k,0], tsne_results[k,1]))

			filename = 'tSNE_'+ling+'_perp='+str(perplexity)+'step='+str(steps)+'lr='+str(lr)+'ang='+str(angle)+'init='+init+'pca='+str(pca)

			plt.savefig(filename + '.pdf')
			np.save(filename, tsne_results)




