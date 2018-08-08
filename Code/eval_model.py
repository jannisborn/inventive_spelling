import warnings
warnings.filterwarnings("ignore",category=FutureWarning)
import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt 
import os, sys, time, argparse
import utils
from utils import acc_new 
from bLSTM import bLSTM


"""
Script to evaluate performance of a model

"""





if __name__ == '__main__':

	parser = argparse.ArgumentParser()

	parser.add_argument('--dataset', default='celex', type=str,
						help='The dataset on which the model was trained, from {celex, childlex, celex_all}')
	parser.add_argument('--learn_type', default='normal', type=str,
						help='The used learning paradigm. Choose from {normal, lds}.')
	parser.add_argument('--task', default='write', type=str,
						help="The task the model solved. Choose from {write, read}.")
	parser.add_argument('--id', default=0, type=int,
						help='The ID of the model that should be examined. Per default 0')
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
		self.epochs = args.epochs - 1

		# Receives the path to the folder of a stored model
		self.root_local = os.path.expanduser("~")+'/Desktop/LDS_Data/'
		self.path = self.root_local + 'TrainedModels/' + self.dataset + '/' + self.learn_type + '_run_' + str(self.id)

		# Set Accuracy object
		self.acc_object = acc_new()
		self.acc_object.accuracy()

		# Retrieve relevant data
		self.retrieve_model_args()
		self.set_hyperparams()

		self.show_mistakes('train')
		print("Training mistakes saved.")
		self.show_mistakes('test')
		#self.predict_input()

		self.plot_pca(mode='input')
		self.plot_pca(mode='output')

		self.plot_tsne(mode='input')
		self.plot_tsne(mode='output')




	def retrieve_model_args(self):
		"""
		Retrieves the hyperparameters of the trained bLSTM (#layers, #nodes, learning rate etc.) which is saved in meta_tags.csv
		"""
		import pandas as pd

		df = pd.read_csv(self.path+'/test_tube_data/version_0/meta_tags.csv')
		raw_args = df['value'].values.tolist()

		self.model_args_write = []
		self.model_args_read = []
		types = ['i','i','i','i','i','i','i','i','i','s','s','b','s','f','s','f','s','b','b','i','i','b','f','l','l'] # ends with test_indices
		for ind,raw_arg in enumerate(raw_args):
			if types[ind] == 'i':
				self.model_args_write.append(int(raw_arg))
			elif types[ind] == 's':
				self.model_args_write.append(str(raw_arg))
			elif types[ind] == 'b':
				self.model_args_write.append(raw_arg==True)  
			elif types[ind] == 'f':
				self.model_args_write.append(float(raw_arg))
			elif types[ind] == 'l':
				str_inds = list(raw_arg)
				self.model_args_write.append(self.join_inds(str_inds))
		
		# The model parameter ordering refers to the writing module. If reading model should be rebuilt, in- and output are flipped
		if self.task == 'read':

			# Flip input and output sequence length
			self.model_args_read.append(self.model_args_write[1]) 
			self.model_args_read.append(self.model_args_write[0])

			# Flip input and output dict sizes
			self.model_args_read.append(self.model_args_write[3]) 
			self.model_args_read.append(self.model_args_write[2])

			# Next arguments are identical for both models
			self.model_args_read.extend(self.model_args_write[4:9])

			# In reading learn type is always normal, set reading property and set mas=500 (dummy)
			self.model_args_read.extend(['normal','read',500])
			self.model_args_read.append(self.model_args_write[-2])
			self.model_args_read.append(self.model_args_write[-1])




	def join_inds(self,str_inds):
		""" 
		Helper method to handle str data from model hyperparameter csv file
		"""

		number_string = ''.join(str_inds)
		number_string_list = number_string.split(",")
		# Remove square brackets
		number_string_list[0] = number_string_list[0][1:] 
		number_string_list[-1] = number_string_list[-1][:-1] 

		return list(map(int,number_string_list))




	def set_hyperparams(self):
		"""
		Sets hyperparameter for the class, based on the user-defined specs (e.g. args.task)

		E.g. it loads the data, retrieves the dictionary to map characters 
		"""

		path = self.root_local + 'data/'
		data = np.load(path + self.dataset + '.npz')

		# Load data
		self.inputs = data['phons'] if self.task == 'write' else data['words']
		self.targets = data['words'] if self.task == 'write' else data['phons']

		# Load dictionaries
		self.input_dict = {key:data['phon_dict'].item().get(key) for key in data['phon_dict'].item()} if self.task == 'write' else {key:data['word_dict'].item().get(key) for key in data['word_dict'].item()}
		self.output_dict = {key:data['word_dict'].item().get(key) for key in data['word_dict'].item()} if self.task == 'write' else  {key:data['phon_dict'].item().get(key) for key in data['phon_dict'].item()}
		self.input_dict_rev = dict(zip(self.input_dict.values(), self.input_dict.keys()))
		self.output_dict_rev = dict(zip(self.output_dict.values(), self.output_dict.keys()))

		print("INPUT DICT", self.input_dict)
		print("OUTPUT DICT", self.output_dict)


		self.inp_seq_nat = 'phonetic' if args.task == 'write' else 'orthografic' # To read in a type of sequence
		self.inp_seq_human = 'spoken' if args.task == 'write' else 'written'
		self.gerund = 'writing' if self.task == 'write' else 'reading'
		self.out_seq_len = self.model_args_write[1] if args.task == 'write' else self.model_args_read[1]



	def predict_input(self):
		"""
		Use this method for command line interaction with the model (showing its predictions to user-specified input words/phonemes).
		Leave method with pressing <SPACE>
		"""

		loop = True

		with tf.Session() as sess:

			# Restore model
			#print(self.path)
			#saver = tf.train.Saver(tf.trainable_variables())
			#saver.restore(sess,tf.train.latest_checkpoint(self.path))
			#sess.run(tf.global_variables_initializer())
			saver = tf.train.import_meta_graph(self.path+'/my_test_model-'+str(self.epochs)+'.meta')
			saver.restore(sess,tf.train.latest_checkpoint(self.path+'/./'))


			#variables_names = [v.name for v in tf.trainable_variables()]
			#values = sess.run(variables_names)
			#for v in variables_names:
			#	print(v)

			graph = tf.get_default_graph()
			keep_prob = graph.get_tensor_by_name(self.gerund+'/keep_prob:0')
			inputs = graph.get_tensor_by_name(self.gerund+'/input:0')
			outputs = graph.get_tensor_by_name(self.gerund+'/output:0')
			logits = graph.get_tensor_by_name(self.gerund+'/decoding_'+args.task+'/logits:0')

			while loop:

				word = input("Please insert a " + self.inp_seq_nat + " sequence: ")

				if word == ' ':
					loop = False
					break

				word_num = self.prepare_sequence(word)
				print(word_num.shape[1],word_num)
				dec_input = np.zeros([1,1]) + self.output_dict['<GO>']

				for k in range(self.out_seq_len):
					pred = sess.run(logits, feed_dict={keep_prob:1.0, inputs:word_num, outputs:dec_input})
					char = pred[:,-1].argmax(axis=-1)
					dec_input = np.hstack([dec_input, char[:,None]]) # Identical to np.expand_dims(char,1)
				print(dec_input.shape[1],dec_input)

				output = ''.join([self.output_dict_rev[num] if self.output_dict_rev[num]!='<PAD>' else '' for ind,num in enumerate(dec_input[0,1:])])
				print("The ", self.inp_seq_nat, " sequence ", word, "  =>  ", output, ' num ', dec_input[0,1:])



	def prepare_sequence(self,word):
		"""
		Prepares a user-inserted word such that it can be feed into the model

		Parameters:
		--------------
		WORD 		{str}		Can be an orthographic word (if task = read) or a phonetic word (if task = write)
		"""
		
		# Error handling
		if any(char.isdigit() for char in word):
			raise TypeError("Please insert a string that contains no numerical values.")

		l = self.model_args_write[0] if args.task == 'write' else self.model_args_read[0] # length of the input sequence 
		phon_word_num = [self.input_dict[word[-k]] if k<=len(word) else self.input_dict['<PAD>'] for k in range(l,0,-1)]

		return np.expand_dims(phon_word_num, axis=0)



	def show_mistakes(self,mode='test'):
		"""
		Show the mistakes of the model on training or testing data and saves the mistakes to a .txt file

		Parameters:
		-----------
		MODE 	{str} either train or test
		"""

		# Retrieve indices of samples the model is tested on
		indices = self.model_args_write[23] if mode=='train' else self.model_args_write[24] # Indices are either train or test indices
		tested_inputs = self.inputs[indices]
		tested_targets = self.targets[indices]

		with tf.Session() as sess: 

			# Restore the model
			saver = tf.train.import_meta_graph(self.path + '/my_test_model-'+str(args.epochs-1)+'.meta')
			saver.restore(sess,tf.train.latest_checkpoint(self.path+'/./'))


			# Retrieve model variables
			keep_prob = graph.get_tensor_by_name(self.gerund+'/keep_prob:0')
			inputs = graph.get_tensor_by_name(self.gerund+'/input:0')
			outputs = graph.get_tensor_by_name(self.gerund+'/output:0')
			logits = graph.get_tensor_by_name(self.gerund+'/decoding_'+self.task+'/logits:0')

			# Prepare model evaluation
			dec_input = np.zeros((len(tested_inputs), 1)) + self.out_dict['<GO>']   # len(tested_inputs) = #tested samples
			for i in range(tested_targets.shape[1]-1): # output sequence has length of target[1] since [0] is batch_size, -1 since <GO> is ignored

			    test_logits = sess.run(logits, feed_dict={keep_prob:1.0, inputs:tested_inputs[:,1:],outputs:dec_input})
			    prediction = test_logits[:,-1].argmax(axis=-1)
			    dec_input = np.hstack([dec_input, prediction[:,None]])

			# Evaluate performance
			fullPred, fullTarg = utils.accuracy_prepare(dec_input[:,1:], tested_targets[:,1:],self.out_dict, mode='test')
			dists, tokenAcc = sess.run([self.acc_object.dists, self.acc_object.token_acc], feed_dict={self.acc_object.fullPred:fullPred, self.acc_object.fullTarg: fullTarg})
			wordAcc  = np.count_nonzero(dists==0) / len(dists) 
			print(self.gerund.upper()+ ' - Accuracy on test set is for tokens{:>6.3f} and for words {:>6.3f}'.format(tokenAcc, wordAcc))



			print('\n',"Now printing the mistakes on the ", mode, " dataset")
			file = open(self.path+'/mistakes_'+mode+'_data.txt','w')

			for ind,pred in enumerate(dec_input[:,1:]):
				if any(pred != tested_targets[ind,1:]):

					inp_str = ''.join([self.input_dict_rev[k] if self.input_dict_rev[k] != '<PAD>' and self.input_dict_rev[k] != '<GO>'  else '' for k in tested_inputs[ind,:]])
					out_str = ''.join([self.output_dict_rev[k] if k!=0 and self.output_dict_rev[k] != '<PAD>' else '' for k in pred])
					tar_str = ''.join([self.output_dict_rev[k] if self.output_dict_rev[k] != '<PAD>' else '' for k in tested_targets[ind,1:]])

					print("The ", inp_seq, " sequence ", inp_str , "  =>  ", out_str, ' instead of ', tar_str, file=file)
			print("Amount of samples in dataset is ", str(ind))
			file.close()






			"""
			saver = tf.train.import_meta_graph(self.path+'/my_test_model-'+str(self.epochs)+'.meta')
			saver.restore(sess,tf.train.latest_checkpoint(self.path+'/./'))


			#variables_names = [v.name for v in tf.trainable_variables()]
			#values = sess.run(variables_names)
			#for v in variables_names:
			#	print(v)

			graph = tf.get_default_graph()
			keep_prob = graph.get_tensor_by_name('reading/keep_prob:0')
			inputs = graph.get_tensor_by_name('reading/input:0')
			outputs = graph.get_tensor_by_name('reading/output:0')
			logits = graph.get_tensor_by_name('reading/decoding_read/logits:0')

			# Iterate over dataset and print all wrong predictions
			tested_inputs = self.inputs[self.indices]
			tested_labels = self.targets[self.indices]
			dec_input = np.zeros((len(tested_inputs),1)) + self.output_dict['<GO>']

			print("tested input ", tested_inputs[1,:])

			# Classify
			for k in range(self.model_args_read[1]): # Length of output sequence

				pred = sess.run(logits, feed_dict={keep_prob:1.0, inputs:tested_inputs[:,1:], outputs:dec_input})
				predictions = pred[:,-1].argmax(axis=-1)
				dec_input = np.hstack([dec_input, predictions[:,None]])


			fullPred, fullTarg = utils.accuracy_prepare(dec_input[:,1:], tested_labels[:,1:], self.output_dict,mode='test')
			dists, token_acc = sess.run([self.acc_object.dists, self.acc_object.token_acc], feed_dict={self.acc_object.fullPred:fullPred, self.acc_object.fullTarg: fullTarg})
			word_acc  = np.count_nonzero(dists==0) / len(dists) 

			print('Accuracy on {:5s} set is for tokens{:>6.3f} and for words {:>6.3f}'.format(mode,token_acc, word_acc))
			"""



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

		with tf.Session() as sess:

			# Restore model
			saver = tf.train.import_meta_graph(self.path+'/my_test_model-'+str(self.epochs)+'.meta')
			saver.restore(sess,tf.train.latest_checkpoint(self.path+'/./'))
			graph = tf.get_default_graph()


			keep_prob = graph.get_tensor_by_name('reading/keep_prob:0')
			inputs = graph.get_tensor_by_name('reading/input:0')
			outputs = graph.get_tensor_by_name('reading/output:0')
			logits = graph.get_tensor_by_name('reading/decoding_read/logits:0')



			if mode=='input':
				#weight_vectors = self.net.input_embedding.eval()
				weight_vectors = sess.run(graph.get_tensor_by_name('reading/encoding_read/enc_embedding:0'))
				print(weight_vectors)
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

			print("The explained variance of the first", n_comp, 'PCs is (in %):', np.round(100*np.sum(pca.explained_variance_ratio_),3))    
    

			if plot:


				fig = plt.figure(figsize = (8,8))
				ax = fig.add_subplot(1,1,1) 
				ax.set_xlabel('Principal Component 1', fontsize = 15)
				ax.set_ylabel('Principal Component 2', fontsize = 15)
				ax.set_title(['Projection of the '+ling+' vectors on the first 2 PCs.'], fontsize = 20)
				ax.scatter(pcs[:,0], pcs[:,1],s=2)
				ax.spines['right'].set_visible(False)
				ax.spines['top'].set_visible(False)
				print(len(pcs))
				for k in range(len(pcs)):
					# in output dict 0 is not used as key
					# will be obsolete after proper retraining
					if mode=='output' and k<28:
						
						ax.annotate(dic[k+1],(pcs[k,0], pcs[k,1]))  
					else:
						ax.annotate(dic[k],(pcs[k,0], pcs[k,1]))

				plt.savefig(self.path+"/PCA_"+ling+"_Results.pdf")

				np.save(self.path+"/PCA_"+ling+"_Results", pcs)

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


			if perplexity >= weight_vectors.shape[0]:
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
			for k in range(len(tsne_results)):
			    # in output dict 0 is not used as key
				# will be obsolete after proper retraining
				if mode=='output' and k<28:
					
					ax.annotate(dic[k+1],(tsne_results[k,0], tsne_results[k,1]))  
				else:
					ax.annotate(dic[k],(tsne_results[k,0], tsne_results[k,1]))  

			filename = self.path+'/tSNE_'+ling+'_perp='+str(perplexity)+'step='+str(steps)+'lr='+str(lr)+'ang='+str(angle)+'init='+init+'pca='+str(pca)

			plt.savefig(filename + '.pdf')
			np.save(filename, tsne_results)



eva = evaluation(args)
