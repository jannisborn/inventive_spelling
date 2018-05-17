import warnings
warnings.filterwarnings("ignore",category=FutureWarning)


import tensorflow as tf 
import numpy as np 

from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple, DropoutWrapper
from tensorflow.contrib.rnn import stack_bidirectional_dynamic_rnn as bi_rnn



# Class with the model. Only forward, no Loss etc.	


# Model implementation:
# An Encoder-Decoder bidirectional Long-Short-Term-Memory Recurent Neural Network Model using sequence_loss and RMSPropOptimizer

class bLSTM(object):

	def __init__(self, input_seq_length, output_seq_length, input_dict_size, num_classes, input_embed_size, output_embed_size, num_layers, num_LSTM_cells, batch_size, 
		learn_type, task, print_ratio=False, optimization='RMSProp', learning_rate=1e-3, LSTM_initializer=None, momentum=0.01, activation_fn=None, bidirectional=True):

		# Task dependent hyperparamter
		self.input_seq_length = input_seq_length        # How long is the input sequence (all have equal length, due to padding)
		self.output_seq_length = output_seq_length		# How long is output sequence (also equal length)
		self.input_dict_size = input_dict_size + 1		# Cardinality of input alphabet 
		self.num_classes = num_classes + 1				# Cardinality of output alphabet (+1 so 0 does not need to be covered since it causes trouble for tf.edit_dist)
		self.learn_type = learn_type					# {'normal', 'lds'} specifies the training regime for the reading.
		self.task = task        						# {'write', 'read'}

		# Network dependent hyperparameter
		self.bidirectional = bidirectional 				# Basic unit of encoder. self.bidirectional is boolean (True by default)
		self.input_embed_size = input_embed_size 		# Feature space dimensionality for individual tokens (one feature vector per possible input character)
		self.output_embed_size = output_embed_size 		# Feature space dimensionality for individual tokens (one feature vector per possible input character)
		self.num_LSTM_cells = num_LSTM_cells			# How many LSTM cells per layer
		self.num_layers = num_layers					# Amount of b-LSTM layers in encoder and LSTM-layers in decoder (default is 1)
		self.LSTM_initializer = LSTM_initializer		# Weight initialization for the LSTM cells (None by default), alternative: tf.contrib.layers.xavier_initializer() or
																#  tf.random_uniform_initializer(-0.1, 0.1)
		self.activation_fn = activation_fn				# AF used in fully connected output layer (None by default), options are:
																# tf.nn.relu, tf.nn.selu, tf.nn.sigmoid, tf.nn.relu6, tf.nn.tanh
		self.batch_size = batch_size					# Batch size
		self.learning_rate = learning_rate				# Learning rate (0.001 by default)
		self.momentum = momentum 						# Only applied in case the momentum optimizer is used (0.01 by default)
		self.optimization = optimization				# Set the optimization technique. Choose from 'RMSProp' (default), 'GD', 'Momentum', 'Adam', 'Adadelta', 'Adagrad'

		# Output dependent hyperparameter
		self.print_ratio = print_ratio					# {bool}, optional, False per default. Only applies if learn_type='lds'. Decides whether the ratio of words
														# that were orthographically incorrect, but accepted from a LdS teacher should be printed.

		# Inferred parameter
		self.decoder_lengths = self.output_seq_length * tf.ones(self.batch_size,dtype=tf.int32) # Currently not needed, only when sequences would have variable length

		# String to function conversion
		self.convert_string_to_functions()


		# Placeholder
		self.inputs = tf.placeholder(tf.int32 , (None,self.input_seq_length),'input')
		self.outputs = tf.placeholder(tf.int32 , (None,None),'output')
		self.targets = tf.placeholder(tf.int32 , (None,None),'targets')
		self.alternative_targets = tf.placeholder(tf.int8, (None,None,None),'alternative_targets') # Orthographically incorrect, but accepted spellings: bs x seq_len x max_alt_targs
		self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')			# Dropout parameter. Determines what ratio of neurons is used (all per default)
		#self.pred_seq_len = tf.placeholder(tf.int32, name='predicted_seq_len')

		self.exe = False



	def convert_string_to_functions(self):

		# Mapping parsed inputs to functions.
		Function_Map = {'None': None ,  None:None,        'Xavier': tf.contrib.layers.xavier_initializer(),    'ReLU': tf.nn.relu,  
                'ReLU6': tf.nn.relu6,   'SeLU': tf.nn.selu,     'Tanh':tf.nn.tanh,           'Sigmoid':tf.sigmoid,
                'RMSProp': tf.train.RMSPropOptimizer, 'GD':tf.train.GradientDescentOptimizer , 'Momentum':tf.train.MomentumOptimizer , 
                'Adam': tf.train.AdamOptimizer, 'Adadelta': tf.train.AdadeltaOptimizer, 'Adagrad': tf.train.AdagradOptimizer ,
                'Uni': tf.random_uniform_initializer(-0.1, 0.1), 'Trunc': tf.truncated_normal_initializer(mean=0.0, stddev=0.1), 
                 }

		# Set the model hyperparameter functions with Function_Map
	    # Converting the strings to the actual functions.
		self.activation_fn = Function_Map[self.activation_fn]
		self.LSTM_initializer = Function_Map[self.LSTM_initializer]


	def forward(self):		

		# Encoder
		with tf.variable_scope("encoding_"+ self.task) as encoding_scope:

			self.input_embedding = tf.Variable(tf.random_uniform((self.input_dict_size, self.input_embed_size), -1.0, 1.0), name='enc_embedding')
			input_embed = tf.nn.embedding_lookup(self.input_embedding, self.inputs)


			if self.bidirectional:

				# Define LSTM cells
				enc_fw_cells = [DropoutWrapper(LSTMCell(self.num_LSTM_cells,initializer=self.LSTM_initializer),input_keep_prob=self.keep_prob) for layer in range(self.num_layers)]
				enc_bw_cells = [DropoutWrapper(LSTMCell(self.num_LSTM_cells,initializer=self.LSTM_initializer),input_keep_prob=self.keep_prob) for layer in range(self.num_layers)]


				# Use the LSTM cells bidirectionally (look forward and backward at input sequence)
				( self.enc_output , enc_fw_final, enc_bw_final) = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(cells_fw=enc_fw_cells, cells_bw=enc_bw_cells, 
				                                                   inputs=input_embed, dtype=tf.float32)

				# Concatenate results
				for k in range(self.num_layers):
					if isinstance(enc_fw_final[k],LSTMStateTuple):
						if k == 0:
							enc_fin_c = tf.concat((enc_fw_final[k].c, enc_bw_final[k].c), 1)
							enc_fin_h = tf.concat((enc_fw_final[k].h, enc_bw_final[k].h), 1)
						else:
							enc_fin_c = tf.concat((enc_fin_c, enc_fw_final[k].c, enc_bw_final[k].c), 1)
							enc_fin_h = tf.concat((enc_fin_h, enc_fw_final[k].h, enc_bw_final[k].h), 1)
					elif isinstance(enc_fw_final[k], tf.Tensor):
						if k == 0:
							enc_state = tf.concat((enc_fw_final[k], enc_bw_final[k]), 1)
					else:
						enc_state = tf.concat((enc_state, enc_fw_final[k], enc_bw_final[k]), 1)
				self.enc_last_state = tf.contrib.rnn.LSTMStateTuple(c=enc_fin_c, h=enc_fin_h)


			else:

				# Define LSTM cells
				enc_cells = [DropoutWrapper(LSTMCell(self.num_LSTM_cells,initializer=self.LSTM_initializer), input_keep_prob=keep_prob) for layer in range(self.num_layers)]
				enc_multi_cell = tf.nn.rnn_cell.MultiRNNCell(enc_cells)
				self.enc_output, self.enc_last_state = tf.nn.dynamic_rnn(enc_cells, inputs=input_embed, dtype=tf.float32)

		# Decoder
		with tf.variable_scope("decoding_"+self.task) as decoding_scope:

			self.output_embedding = tf.Variable(tf.random_uniform((self.num_classes, self.output_embed_size), -1.0, 1.0), name='dec_embedding')
			output_embed = tf.nn.embedding_lookup(self.output_embedding, self.outputs)

			if self.bidirectional:

				dec_cells = tf.contrib.rnn.LSTMCell(2*self.num_layers*self.num_LSTM_cells,initializer=self.LSTM_initializer)


			else:
				dec_cells_list = [tf.contrib.rnn.LSTMCell(self.num_LSTM_cells,initializer=self.LSTM_initializer) for _ in range(self.num_layers)]
				dec_cells = tf.nn.rnn_cell.MultiRNNCell(dec_cells_list)

			self.dec_outputs, _ = tf.nn.dynamic_rnn(dec_cells, inputs=output_embed, initial_state=self.enc_last_state)


			# Fully connected layer of the decoder outputs to the predictions
			self.all_logits = tf.contrib.layers.fully_connected(self.dec_outputs, num_outputs=self.num_classes, activation_fn=self.activation_fn) 
			self.logits = tf.contrib.layers.dropout(self.all_logits, self.keep_prob) 
			self.logits = tf.identity(self.logits, name='logits')



	def backward(self):

		with tf.name_scope("optimization_"+ self.task):

			# Loss function
			if self.learn_type == 'normal':
				self.loss = tf.contrib.seq2seq.sequence_loss(self.logits, self.targets, tf.ones([self.batch_size, self.output_seq_length]))
			elif self.learn_type == 'lds':
				print(self.exe, "EXE")
				self.loss_lds, self.read_inps = tf.contrib.seq2seq.sequence_loss_lds(self.logits, self.targets, 
					tf.ones([self.batch_size, self.output_seq_length]), self.alternative_targets, self.exe)
			else:
				raise ValueError("Unspecified learning regime.")


			# Optimizer
			if self.optimization == 'GD':
				self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)
			elif self.optimization == 'Momentum':
				print("Learning rate and momentum are set.")
				self.optimizer = tf.train.MomentumOptimizer(self.learning_rate,self.momentum).minimize(self.loss)
			elif self.optimization == 'Adam':
				print("No learning rate to set.")
				self.optimizer = tf.train.AdamOptimizer().minimize(self.loss)
			elif self.optimization == 'Adadelta':
				print("No learning rate to set.")
				self.optimizer = tf.train.AdadeltaOptimizer().minimize(self.loss)
			elif self.optimization == 'Adagrad':
				self.optimizer = tf.train.AdagradOptimizer(self.learning_rate).minimize(self.loss)
			elif self.optimization == 'RMSProp':
				self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)


				# Clip gradients?










