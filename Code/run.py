    
# Write a .txt file that reads in all the input arguments.
# tqdm?




###################     IMPORTING      ######################
import warnings
warnings.filterwarnings("ignore",category=FutureWarning)

import sys, os, warnings, argparse, time
import random, shutil
import numpy as np
import tensorflow as tf
import h5py as h5py
import matplotlib.pyplot as plt

# Import functions from some modules
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from test_tube import Experiment
from time import time

# Import my files
from utils import acc_new
import utils
from bLSTM import bLSTM
#from eval_model import evaluation


sys.path.append('../')




################## READ IN ARGUMENTS  #################################


# Set possible arguments for the model
if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Storage and saving hyperparameter
    parser.add_argument('--log_dir', default=1, type=int,
                        help='Options where to store log (see below...)')
    parser.add_argument('--file_name', default='run_', type=str,
                        help='The filename for the current run, if Empty lets utils naming convention name file, '
                             'if not empty overwrites naming convention')
    parser.add_argument('--extra', default = '', type=str,
                        help='Extra string to add to end of standard naming convention')
    parser.add_argument('--run_id', default=0, type=int,
                        help='The run id of the current run')
    parser.add_argument('--max_outputs', default=4, type=int,
                        help='Max number of samples to save in summaries')
    parser.add_argument('--restore', default=False, type=bool, 
                        help='Restore a pretrained model or initialize a new one (default).')
    parser.add_argument('--save_model', default=200, type=int,
                        help='Frequency of iterations before model is saved and stored.')

    # Task hyperparameter
    parser.add_argument('--task', default='fibel', type=str,
                        help="Sets the task to solve, default is 'TIMIT_P2G', alternatives are 'Dates', 'TIMIT_G2P' "
                        " and later on more...")
    parser.add_argument('--learn_type', default='normal', type=str,
                        help="Determines the training regime. Choose from set {'normal', 'lds'}.")
    parser.add_argument('--reading', default=True, type=bool,
                        help="Specifies whether reading task is also accomplished. Default is False. ")


    # Training and recording hyperparameter
    parser.add_argument('--epochs', default=1000, type=int,
                        help='The number of epochs to train on')
    parser.add_argument('--print_step', default=10, type=int,
                        help='Record training & test accuracy after every n epochs')
    parser.add_argument('--batch_size', default=1500, type=int,
                        help='The batch size for training')
    parser.add_argument('--seed', default=42, type=int,
                        help='Seed for the random number generator')
    parser.add_argument('--print_ratio', default=False, type=bool,
                        help='For LdS training regime, whether ratio of incorrect but accepted words should be printed.')
    parser.add_argument('--show_plot', default=False, type=bool,
                        help='Specifies whether Accuracy plots are shown at end of training. Do only if machine you run on has GUI')

    # Model hyperparameter
    parser.add_argument('--learning_rate', default=1e-03, type=float,
                        help='The learning rate of the optimizer')
    parser.add_argument('--input_embed_size', default=150, type=int,
                        help='The feature space dimensionality for the input characters')
    parser.add_argument('--output_embed_size', default=150, type=int,
                        help='The feature space dimensionality for the output characters')
    parser.add_argument('--num_nodes', default=128, type=int,
                        help='The number of LSTM nodes per layer in both encoder and decoder')
    parser.add_argument('--num_layers', default=2, type=int,
                        help='The number of layers in both encoder and decoder')
    parser.add_argument('--optimization',default='RMSProp', type=str, 
                        help="The optimizer used in the model. 'RMSProp' as default, give as string, alternatives: 'GD', "
                        "'Momentum', 'Adam', 'Adadelta', 'Adagrad' ")
    parser.add_argument('--LSTM_initializer', default='None', type=str,
                        help="The weight initializer for the LSTM cells. None as default, give as string, alternative 'Xavier' ")
    parser.add_argument('--activation_fn', default='None', type=str,
                        help="The weight initializer for the LSTM cells. None as default, give as string, alternatives: 'ReLU', " 
                        "'ReLu6', 'SeLU', 'Sigmoid', 'Tanh' ")
    parser.add_argument('--momentum', default=0.01, type=float,
                        help="The momentum parameter. Only used in case the momentum optimizer is used.")
    parser.add_argument('--bidirectional', default=True, type=bool,
                        help="Basic unit of encoder is bidirectional by default. Set to False to use regular LSTM units.")
    parser.add_argument('--dropout', default=1.0, type=float,
                        help="Dropout probability of neurons during training.")
    parser.add_argument('--test_size', default=0.05, type=float,
                        help="Percentage of dataset hold back for testing.")




    parser.add_argument('--gpu_options', default=3, type=int,
                        help='GPU options for tensorflow, 0 to allow growth and'
                             '1 to set gpu fraction limit and '
                             '2 for no GPU limiting'
                             '3 for no GPU training')
    parser.add_argument('--gpu_fraction', default=0.3, type=float,
                        help='If GPU options is set to 1, sets GPU fraction')


    theta_min = 20
    theta_max = 20



    # Parse input arguments
    args = parser.parse_args()


##########################    PROCESS ARGUMENTS     ####################################

    if args.log_dir == 0:
        log_dir = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), 'logs')
    elif args.log_dir == 1:
        log_dir = "../../"

    print('\n You have choosen the following options: \n ', args, '\n')
    file_name = args.file_name + str(args.run_id)
    save_path = os.path.join(log_dir,'Models', args.task, args.learn_type + '_' + file_name)
    test_tube = save_path

    # if save_path does not exist, create one
    while os.path.exists(save_path):
        print('This model name already exists as {}'.format(save_path))
        print('Increment run_id by 1')
        args.run_id += 1
        file_name = "run_" + str(args.run_id)
        save_path = os.path.join(log_dir,'Models', args.task, args.learn_type + '_' + file_name)
        test_tube = save_path

    # if save_path does not exist, create one
    if not os.path.exists(save_path):
        os.makedirs(save_path)



    # setting the gpu options if any
    if args.gpu_options == 0:
        sess = get_growth_allowed_session()
    elif args.gpu_options == 1:
        sess = get_limited_gpu_session(gpu_fraction=args.gpu_fraction)
    else:
        sess = tf.Session()

    # setting a random seed for reproducibility
    np.random.seed(args.seed)




    # LOAD DATA

    if args.task == 'Date':

        # Retrieve date data from tutorial
        ((inputs, targets) , (dict_char2num_x, dict_char2num_y)) = utils.date_dataset(args.seed) 
                
    elif args.task == 'timit_p2g':

        # Writing with TIMIT
        ((inputs, targets) , (dict_char2num_x, dict_char2num_y)) = utils.TIMIT_P2G() 

    elif args.task == 'timit_g2p':

        # Reading with TIMIT
        ((inputs, targets) , (dict_char2num_x, dict_char2num_y)) = utils.TIMIT_G2P() 

    elif args.task == 'bas_g2p_c':
        path = '/Users/jannisborn/Desktop/LDS_Data/BAS_SprecherInnen'
        ((inputs, targets) , (dict_char2num_x, dict_char2num_y)) = utils.BAS_G2P_create(path)


    elif args.task == 'bas_p2g_c':
        # Follows
        path = '/Users/jannisborn/Desktop/LDS_Data/BAS_SprecherInnen'
        ((inputs, targets) , (dict_char2num_x, dict_char2num_y)) = utils.BAS_P2G_create(path)

    elif args.task == 'bas_g2p_r':
        ((inputs, targets) , (dict_char2num_x, dict_char2num_y)) = utils.BAS_G2P_retrieve()

    elif args.task == 'bas_p2g_r':
        ((inputs, targets) , (dict_char2num_x, dict_char2num_y)) = utils.BAS_P2G_retrieve()

    elif args.task == 'celex' and args.learn_type == 'normal':
        ((inputs, targets) , (dict_char2num_x, dict_char2num_y)) = utils.celex_retrieve(args.learn_type)
    
    elif args.task == 'celex' and args.learn_type == 'lds':
        ((inputs, targets) , (dict_char2num_x, dict_char2num_y), alt_targets) = utils.celex_retrieve(args.learn_type)

    elif args.task == 'bas':
        ((inputs, targets) , (dict_char2num_x, dict_char2num_y)) = utils.BAS_P2G_retrieve()

    #elif args.task == 'childlex' and args.learn_type == 'normal':
    #    ((inputs, targets) , (dict_char2num_x, dict_char2num_y)) = utils.childlex_retrieve(args.learn_type)

    elif args.task == 'childlex':
        ((inputs, targets) , (dict_char2num_x, dict_char2num_y), alt_targets) = utils.childlex_retrieve()
        mas = 100

    #elif args.task == 'fibel' and args.learn_type == 'normal':
    #    ((inputs, targets) , (dict_char2num_x, dict_char2num_y)) = utils.fibel_retrieve(args.learn_type)

    elif args.task == 'fibel':
        ((inputs, targets) , (dict_char2num_x, dict_char2num_y), alt_targets) = utils.fibel_retrieve()
        lektions_inds = [9,14,20,28,36,46,58,77,99,121,154,174]
        mas = 810


    """
        # Very different training regime and thus done within this if/else case.

        x_dict_size, num_classes, x_seq_length, y_seq_length, dict_num2char_x, dict_num2char_y = utils.set_model_params(inputs, targets, dict_char2num_x, dict_char2num_y)

        with tf.variable_scope('writing'):
            model_write = bLSTM(x_seq_length, y_seq_length, x_dict_size, num_classes, args.input_embed_size, args.output_embed_size, args.num_layers, args.num_nodes, args.batch_size,
                args.learn_type, 'write', print_ratio=args.print_ratio, optimization=args.optimization ,learning_rate=args.learning_rate, LSTM_initializer=args.LSTM_initializer, 
                momentum=args.momentum, activation_fn=args.activation_fn, bidirectional=args.bidirectional)
            model_write.forward()
            model_write.backward()
            model_write.exe = True
            saver_write = tf.train.Saver([k for k in tf.global_variables() if k.name.startswith("writing")], max_to_keep=10)



        # Should the reading module be enabled?
        if args.reading:
            with tf.variable_scope('reading'):
                model_read = bLSTM(y_seq_length, x_seq_length, num_classes, x_dict_size, args.input_embed_size, args.output_embed_size, args.num_layers, args.num_nodes,
                    args.batch_size, 'normal', 'read',print_ratio=args.print_ratio, optimization=args.optimization ,learning_rate=args.learning_rate, 
                    LSTM_initializer=args.LSTM_initializer, momentum=args.momentum, activation_fn=args.activation_fn, bidirectional=args.bidirectional)
                # Learn type is always normal, but if regime is lds, then corrupted input may be used.
                model_read.forward()
                model_read.backward()
                saver_read = tf.train.Saver([k for k in tf.global_variables() if k.name.startswith("reading")], max_to_keep=10)
        

        exp = Experiment(name='', save_dir=test_tube)
        # First K arguments are in the same order like the ones to initialize the bLSTM, this simplifies restoring
        exp.add_meta_tags({'inp_len':x_seq_length, 'out_len':y_seq_length, 'x_dict_size':x_dict_size, 'num_classes':num_classes, 'input_embed':args.input_embed_size,
                                'output_embed':args.output_embed_size, 'num_layers':args.num_layers, 'nodes/Layer':args.num_nodes, 'batch_size':args.batch_size, 'learn_type':
                                args.learn_type, 'task': 'write', 'print_ratio':args.print_ratio, 'optimization':str(args.optimization), 'lr': args.learning_rate,
                                'LSTM_initializer':str(args.LSTM_initializer), 'momentum':args.momentum,'ActFctn':str(args.activation_fn), 'bidirectional': args.bidirectional,  
                                 'Write+Read = ': args.reading, 'epochs': args.epochs,  'seed':args.seed,'restored':args.restore, 'dropout':args.dropout, 'train_indices':
                                 indices_train, 'test_indices':indices_test})

        regime = args.learn_type

        # For every Lektion, train k epochs, then test once.
        for k in range(len(lektions_inds)):

            print("Now training Lektion "+str(k+1))

            for e in range(args.epochs):

                print("Epoch nr "+str(e))
                ind = 0 if k==0 else lektions_inds[k-1]
                write_inp = inputs[ind:lektions_inds[k],1:]
                write_out = targets[ind:lektions_inds[k],:-1]
                write_targets = targets[ind:lektions_inds[k],1:]


                if regime == 'normal':

                    _, batch_loss, batch_logits = sess.run([model_write.optimizer, model_write.loss, model_write.logits], feed_dict = 
                                                            {model_write.keep_prob: args.dropout, model_write.inputs: write_inp,model_write.outputs: write_target,
                                                             model_write.targets: write_targets})
                # Needs some more work
                elif regime == 'lds':
                    # Inputs for readings are returned from writing process
                    _, batch_loss, read_inp, batch_logits = sess.run([model_write.optimizer, model_write.loss_lds, model_write.read_inps, model_write.logits], 
                                                            feed_dict = 
                                                            {model_write.keep_prob: args.dropout, model_write.inputs: write_inp,model_write.outputs: write_out,
                                                             model_write.targets: write_targets, model_write.alternative_targets: alternative_targets})

                if args.reading and regime == 'normal':
                    read_inp = write_targets
                    read_out = inputs[ind:lektions_inds[k],:-1]
                    read_targets = write_inp

                    _, batch_loss, batch_logits = sess.run([model_read.optimizer, model_read.loss, model_read.logits], feed_dict = 
                                                        {model_read.keep_prob:args.dropout, model_read.inputs: read_inp, model_read.outputs:read_out,
                                                         model_read.targets:read_targets})
                
                if args.reading and regime == 'lds':
                    # Needs more work
                    read_out = inputs[ind:lektions_inds[k],:-1]
                    read_targets = write_inp

                    _, batch_loss, batch_logits = sess.run([model_read.optimizer, model_read.loss, model_read.logits], feed_dict = 
                                                        {model_read.keep_prob:args.dropout, model_read.inputs: read_inp, model_read.outputs:read_out,
                                                         model_read.targets:read_targets})
                if regime=='lds' and args.epoch//2 == e:
                    regime = 'normal' # change learning regime if half of the epochs are over
                    model_write.learn_type = 'normal'
                    print("Training regime changed to normal")

            # reset regime.
            regime = args.learn_type 
            model_write.learn_type = args.learn_type

        # ------------------------ TESTING -----------------------------
        accs = np.zeros(4,len(lektions_inds))
        for k in range(len(lektions_inds)):
            ind = 0 if k==0 else lektions_inds[k-1]
            write_inp = inputs[ind:lektions_inds[k],1:]
            write_targets = targets[ind:lektions_inds[k],1:]

            write_dec_input = np.zeros((len(write_targets), 1)) + dict_char2num_y['<GO>']
            # Generate character by character (for the entire batch, weirdly)
            for i in range(y_seq_length):
                write_test_logits = sess.run(model_write.logits, feed_dict={model_write.keep_prob:1.0, model_write.inputs:write_inp, model_write.outputs:write_dec_input})
                write_prediction = write_test_logits[:,-1].argmax(axis=-1)
                #print('Loop',test_logits.shape, test_logits[:,-1].shape, prediction.shape)
                write_dec_input = np.hstack([write_dec_input, write_prediction[:,None]])
            #print(dec_input[:,1:].shape, Y_test[:,1:].shape)
            write_oldAcc, write_tokenAcc , write_wordAcc = utils.accuracy(write_dec_input[:,1:], write_targets,dict_char2num_y, mode='test')
            print('WRITING - Accuracy for lektion{:>6.3f} is for tokens{:>6.3f} and for words {:>6.3f}'.format(k+1,write_tokenAcc, write_wordAcc))
            accs[0,k] = write_tokenAcc
            accs[1,k] = write_wordAcc

            # Test READING
            if args.reading:
                read_inp = write_targets
                read_targets = write_inp
                read_dec_input = np.zeros((len(read_targets), 1)) + dict_char2num_x['<GO>']
                # Generate character by character (for the entire batch, weirdly)
                for i in range(x_seq_length):
                    read_test_logits = sess.run(model_read.logits, feed_dict={model_read.keep_prob:1.0, model_read.inputs:read_inp, model_read.outputs:read_dec_input})
                    read_prediction = read_test_logits[:,-1].argmax(axis=-1)
                    #print('Loop',test_logits.shape, test_logits[:,-1].shape, prediction.shape)
                    read_dec_input = np.hstack([read_dec_input, read_prediction[:,None]])
                #print(dec_input[:,1:].shape, Y_test[:,1:].shape)
                read_oldAcc, read_tokenAcc , read_wordAcc = utils.accuracy(read_dec_input[:,1:], read_targets,dict_char2num_x, mode='test')
                print('READING - Accuracy for lektion{:>6.3f} is for tokens{:>6.3f} and for words {:>6.3f}'.format(k+1,read_tokenAcc, read_wordAcc))
                accs[2,k] = read_tokenAcc
                accs[3,k] = read_wordAcc

        saver_write.save(sess, save_path + '/Model_write', global_step=epoch, write_meta_graph=False)
        if args.reading:
            saver_read.save(sess, save_path + '/Model_read', global_step=epoch, write_meta_graph=False)           
                    

        print(" Training done, model_write saved in file: %s" % save_path + ' ' + os.path.abspath(save_path))

        np.savetxt(save_path+'/performance.txt', accs, delimiter=',')   


        print("DONE!")
        sys.exit(0)

    """






    # -------------------------------------------- REGULAR TRAINING SETUP --------------------------------------------------- #

    # Set remaining parameter based on the processed data
    x_dict_size, num_classes, x_seq_length, y_seq_length, dict_num2char_x, dict_num2char_y = utils.set_model_params(inputs, targets, dict_char2num_x, dict_char2num_y)
    print(dict_num2char_x)
    print()
    print(dict_num2char_y)

    # Split data into training and testing
    indices = range(len(inputs))
    
    #if args.learn_type == 'normal':
    #X_train, X_test,Y_train, Y_test, indices_train, indices_test = train_test_split(inputs, targets, indices, test_size=args.test_size, random_state=args.seed)
    #elif args.learn_type == 'lds':
    X_train, X_test,Y_train, Y_test, Y_alt_train_l, Y_alt_test_l, indices_train, indices_test = train_test_split(inputs, targets, alt_targets, indices, test_size=args.test_size, random_state=args.seed)
    max_len = max([len(l) for l in Y_alt_train_l])
    inp_seq_len = len(Y_alt_train_l[1][0])
    Y_alt_train = np.zeros([len(Y_alt_train_l), inp_seq_len, max_len], dtype=np.int8)
    for word_ind in range(len(Y_alt_train_l)):
        for write_ind in range(len(Y_alt_train_l[word_ind])):
            Y_alt_train[word_ind,:,write_ind] = np.array(Y_alt_train_l[word_ind][write_ind],dtype=np.int8)

    max_len = max([len(l) for l in Y_alt_test_l])
    inp_seq_len = len(Y_alt_test_l[1][0])
    Y_alt_test = np.zeros([len(Y_alt_test_l), inp_seq_len, max_len], dtype=np.int8)
    for word_ind in range(len(Y_alt_test_l)):
        for write_ind in range(len(Y_alt_test_l[word_ind])):
            Y_alt_test[word_ind,:,write_ind] = np.array(Y_alt_test_l[word_ind][write_ind],dtype=np.int8)

    print(X_train.shape, Y_train.shape, Y_alt_train.shape,'PRESHAPES')





    ############## PREPARATION FOR TRAINING ##############

    regime = args.learn_type

    with tf.variable_scope('writing'):

        model_write = bLSTM(x_seq_length, y_seq_length, x_dict_size, num_classes, args.input_embed_size, args.output_embed_size, args.num_layers, args.num_nodes, args.batch_size,
            args.learn_type, 'write', mas, print_ratio=args.print_ratio, optimization=args.optimization ,learning_rate=args.learning_rate, LSTM_initializer=args.LSTM_initializer, 
            momentum=args.momentum, activation_fn=args.activation_fn, bidirectional=args.bidirectional)
        model_write.forward()
        model_write.backward()
        model_write.exe = True

        saver_write = tf.train.Saver([k for k in tf.global_variables() if k.name.startswith("writing")], max_to_keep=10)



    # Should the reading module be enabled?
    if args.reading:
        with tf.variable_scope('reading'):
            model_read = bLSTM(y_seq_length, x_seq_length, num_classes, x_dict_size, args.input_embed_size, args.output_embed_size, args.num_layers, args.num_nodes,
                args.batch_size, 'normal', 'read', mas, print_ratio=args.print_ratio, optimization=args.optimization ,learning_rate=args.learning_rate, 
                LSTM_initializer=args.LSTM_initializer, momentum=args.momentum, activation_fn=args.activation_fn, bidirectional=args.bidirectional)
            model_read.forward()
            model_read.backward()
            saver_read = tf.train.Saver([k for k in tf.global_variables() if k.name.startswith("reading")], max_to_keep=10)


    exp = Experiment(name='', save_dir=test_tube)
    # First K arguments are in the same order like the ones to initialize the bLSTM, this simplifies restoring
    exp.add_meta_tags({'inp_len':x_seq_length, 'out_len':y_seq_length, 'x_dict_size':x_dict_size, 'num_classes':num_classes, 'input_embed':args.input_embed_size,
                        'output_embed':args.output_embed_size, 'num_layers':args.num_layers, 'nodes/Layer':args.num_nodes, 'batch_size':args.batch_size, 'learn_type':
                        args.learn_type, 'task': 'write', 'print_ratio':args.print_ratio, 'optimization':str(args.optimization), 'lr': args.learning_rate,
                        'LSTM_initializer':str(args.LSTM_initializer), 'momentum':args.momentum,'ActFctn':str(args.activation_fn), 'bidirectional': args.bidirectional,  
                         'Write+Read = ': args.reading, 'epochs': args.epochs,  'seed':args.seed,'restored':args.restore, 'dropout':args.dropout, 'train_indices':
                         indices_train, 'test_indices':indices_test})

    #saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=None)) # Add ops to save and restore all the variables.
    #sess.run(tf.global_variables_initializer())

    # builds the histogram of the parameter values for viewing in tensorboard
    for var in tf.trainable_variables():
        tf.summary.histogram(var.name, var)


    # builds the scalar summaries of loss and error rate metrics
    #tf.summary.scalar('Seq2Seq Loss', model_write.loss)

    # builds the histogram of GRU activations
    tf.summary.histogram('GRU activations', model_write.all_logits)

    # calculates the total number of parameters in the network
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parameters = 1
        try:
            for dim in shape:
                variable_parameters *= dim.value
        except ValueError:
            variable_parameters += 0
        total_parameters += variable_parameters
    print('{} trainable parameters in the network.'.format(total_parameters))

    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(save_path + '/train', sess.graph)
    test_writer = tf.summary.FileWriter(save_path + '/test')




    if args.restore:
        utils.retrieve_model()
    else:
        # tensor to initialize the variables
        print("BEF")
        init_tensor = tf.global_variables_initializer()
        # initializing the variables
        sess.run(init_tensor)
        print("AF")

    #if args.restore:
    #saver.restore(sess, _model_writePath) #Yes, no need to add ".index"

    num_train_steps, num_val_steps = 0, 0

    if args.reading:
        trainPerf = np.zeros([args.epochs//args.print_step + 1, 6])
        testPerf = np.zeros([args.epochs//args.print_step + 1, 6])
        read_losses = np.zeros((args.epochs,1))
        read_losses_test = np.zeros((args.epochs//args.print_step + 1,1))

    else:
        trainPerf = np.zeros([args.epochs//args.print_step + 1, 3])
        testPerf = np.zeros([args.epochs//args.print_step + 1, 3])
        

    lds_ratios = np.zeros((args.epochs,1))
    corr_ratios = np.zeros((args.epochs,1))
    lds_losses = np.zeros((args.epochs,1))
    reg_losses = np.zeros((args.epochs,1))

    lds_ratios_test = np.zeros((args.epochs//args.print_step + 1,1))
    corr_ratios_test = np.zeros((args.epochs//args.print_step + 1,1))
    lds_losses_test = np.zeros((args.epochs//args.print_step + 1,1))
    reg_losses_test = np.zeros((args.epochs//args.print_step + 1,1))


    # Accuracy object
    acc_object  = acc_new()
    acc_object.accuracy()





    print('\n Starting training \n ')
    for epoch in range(args.epochs):

        print('Epoch ', epoch + 1)
        t = time()    
        # Regular training (do not show performance)
        #if epoch % args.print_step != 0 :

        # Modify this if learn_type is lds: foor loop needs to incorporate alternative targets.
        rats_lds = []
        rats_corr = []
        lds_loss = []
        reg_loss = []
        read_loss = []

        if regime == 'normal':
        
            for k, (write_inp_batch, write_out_batch, write_alt_targs) in enumerate(utils.batch_data(X_train, Y_train, args.batch_size,Y_alt_train)):
            
            # Train Writing
                _, batch_loss, w_batch_logits, loss_lds, rat_lds, rat_corr = sess.run([model_write.optimizer, model_write.loss, model_write.logits, 
                    model_write.loss_lds, model_write.rat_lds, model_write.rat_corr], feed_dict = 
                                                        {model_write.keep_prob: args.dropout, model_write.inputs: write_inp_batch[:, 1:], 
                                                        model_write.outputs: write_out_batch[:, :-1], model_write.targets: write_out_batch[:, 1:],
                                                        model_write.alternative_targets: write_alt_targs[:,1:,:]})
                rats_lds.append(rat_lds)
                rats_corr.append(rat_corr)
                lds_loss.append(loss_lds)
                reg_loss.append(batch_loss)


                if epoch > theta_min and epoch < theta_max:

                    utils.num_to_str(write_inp_batch,w_batch_logits,write_out_batch,write_alt_targs,dict_num2char_x,dict_num2char_y)


                if args.reading:

                    read_inp_batch = write_out_batch
                    read_out_batch = write_inp_batch
                    #read_out_batch = np.concatenate([np.ones([args.batch_size,1],dtype=np.int64) * dict_char2num_x['<GO>'], write_inp_batch],axis=1)

                    _, batch_loss, batch_logits = sess.run([model_read.optimizer, model_read.loss, model_read.logits], feed_dict = 
                                                    {model_read.keep_prob:args.dropout, model_read.inputs: read_inp_batch[:,1:], 
                                                    model_read.outputs:read_out_batch[:,:-1], model_read.targets:read_out_batch[:,1:]})

                    read_loss.append(batch_loss)

        elif regime == 'lds':


            for k, (write_inp_batch, write_out_batch, write_alt_targs) in enumerate(utils.batch_data(X_train, Y_train, args.batch_size, Y_alt_train)):

                _, batch_loss, write_new_targs, rat_lds, rat_corr, batch_loss_reg, w_batch_logits= sess.run([model_write.lds_optimizer, model_write.loss_lds, model_write.read_inps, 
                    model_write.rat_lds, model_write.rat_corr, model_write.loss_reg, model_write.logits], 
                                feed_dict = 
                                                        {model_write.keep_prob:args.dropout, model_write.inputs: write_inp_batch[:,1:], 
                                                        model_write.outputs: write_out_batch[:, :-1], model_write.targets: write_out_batch[:, 1:], 
                                                        model_write.alternative_targets: write_alt_targs[:,1:,:]})


                rats_lds.append(rat_lds)
                rats_corr.append(rat_corr)
                lds_loss.append(batch_loss)
                reg_loss.append(batch_loss_reg)

                if epoch > theta_min and epoch < theta_max:

                    r = utils.num_to_str(write_inp_batch,w_batch_logits,write_out_batch,write_alt_targs,dict_num2char_x,dict_num2char_y)
                    print("Original target was ", write_out_batch[r,:], "New target is ", write_new_targs[r,:])
                    print("Another word: ", write_out_batch[r-1,:], "New target is ", write_new_targs[r-1,:])

                if args.reading:
                    read_inp_batch = write_new_targs
                    read_out_batch = write_inp_batch
                    _, batch_loss, batch_logits = sess.run([model_read.optimizer, model_read.loss, model_read.logits], feed_dict = 
                                                    {model_read.keep_prob:args.dropout, model_read.inputs: read_inp_batch, 
                                                    model_read.outputs:read_out_batch[:,:-1], model_read.targets:read_out_batch[:,1:]})
                    read_loss.append(batch_loss)

                  
        lds_losses[epoch] = sum(lds_loss)/len(lds_loss)
        reg_losses[epoch] = sum(reg_loss)/len(reg_loss)
        lds_ratios[epoch] = sum(rats_lds)/len(rats_lds)
        corr_ratios[epoch] = sum(rats_corr)/len(rats_corr)
        read_losses[epoch] = sum(read_loss)/len(read_loss)

        print("Ratio correct  words: " + str(corr_ratios[epoch])+" and in LdS sense: " + str(lds_ratios[epoch]))
        print("LdS loss is " + str(lds_losses[epoch]) + " while regular loss is" + str(reg_losses[epoch]))


        """
        # Alternative
        dec_input = np.zeros((len(write_inp_batch), 1)) + dict_char2num_y['<GO>']
        # Generate character by character (for the entire batch, weirdly)
        for i in range(y_seq_length):
            targs = write_out_batch[:,1:2+i]
            #print("Does not work", dec_input.shape, targs.shape)
            _, batch_loss, batch_logits = sess.run([model_write.optimizer, model_write.loss, model_write.logits], feed_dict = 
                                                    {model_write.keep_prob:args.dropout, model_write.inputs:write_inp_batch, 
                                                    model_write.outputs:dec_input, model_write.targets:targs, model_write.pred_seq_len:i+1})
            prediction = batch_logits[:,-1].argmax(axis=-1)
            #print('Loop',test_logits.shape, test_logits[:,-1].shape, prediction.shape)
            dec_input = np.hstack([dec_input, prediction[:,None]])
        """
               
                


            
        #else: # Display performance
        if epoch % args.print_step == 0: 
            # ---------------- SHOW TRAINING PERFORMANCE -------------------------
            
            rats_lds = []
            rats_corr = []
            lds_loss = []
            reg_loss = []
            read_loss = []

            # Allocate variables
            write_word_accs = np.zeros(len(X_train)// args.batch_size)
            write_token_accs = np.zeros(len(X_train)// args.batch_size)
            write_old_accs = np.zeros(len(X_train)// args.batch_size)
            #write_word_accs_2 = np.zeros(len(X_train)// args.batch_size)
            #write_token_accs_2 = np.zeros(len(X_train)// args.batch_size)
            #write_old_accs_2 = np.zeros(len(X_train)// args.batch_size)
            write_epoch_loss = 0

            read_word_accs = np.zeros(len(X_train)// args.batch_size)
            read_token_accs = np.zeros(len(X_train)// args.batch_size)
            read_old_accs = np.zeros(len(X_train)// args.batch_size)
            read_epoch_loss = 0
                

            #print("Time it took til initializing: ", time()-t)

            if regime == 'normal':

                for k, (write_inp_batch, write_out_batch,write_alt_targs) in enumerate(utils.batch_data(X_train, Y_train, args.batch_size, Y_alt_train)):
                    _, batch_loss, w_batch_logits, loss_lds, rat_lds, rat_corr = sess.run([model_write.optimizer, model_write.loss, model_write.logits, 
                        model_write.loss_lds, model_write.rat_lds, model_write.rat_corr], feed_dict =
                                                             {model_write.keep_prob:1.0, model_write.inputs: write_inp_batch[:,1:], 
                                                             model_write.outputs: write_out_batch[:, :-1], model_write.targets: write_out_batch[:, 1:],
                                                            model_write.alternative_targets: write_alt_targs[:,1:,:]})

                    #print("Time it took to run one batch for reading: ", time()-t)

                    write_epoch_loss += batch_loss
                    #t=time()
                    #write_old_accs[k], write_token_accs[k] , write_word_accs[k] = utils.accuracy(w_batch_logits, write_out_batch[:,1:], dict_char2num_y)
                    #print("Time for regular accuracy ", time()-t)

                    #t=time()
                    write_old_accs[k], fullPred, fullTarg = utils.accuracy_prepare(w_batch_logits, write_out_batch[:,1:], dict_char2num_y)
                    
                    dists, write_token_accs[k] = sess.run([acc_object.dists, acc_object.token_acc], 
                            feed_dict={acc_object.fullPred:fullPred, acc_object.fullTarg: fullTarg})

                    write_word_accs[k] = np.count_nonzero(dists==0) / len(dists) 
                    #print("Time for new accuracy ", time()-t)


                    rats_lds.append(rat_lds)
                    rats_corr.append(rat_corr)
                    lds_loss.append(loss_lds)
                    reg_loss.append(batch_loss)

                    if epoch > theta_min and epoch < theta_max:
                        utils.num_to_str(write_inp_batch,w_batch_logits,write_out_batch,write_alt_targs,dict_num2char_x,dict_num2char_y)

                    #print("Time it took compute analysis: ", time()-tt)

                    # Test reading
                    if args.reading:
                        read_inp_batch = write_out_batch
                        read_out_batch = write_inp_batch
                        #read_out_batch = np.concatenate([np.ones([args.batch_size,1],dtype=np.int64) * dict_char2num_x['<GO>'], write_inp_batch],axis=1)

                        _, batch_loss, r_batch_logits = sess.run([model_read.optimizer, model_read.loss, model_read.logits], feed_dict =
                                                             {model_read.keep_prob:1.0, model_read.inputs: read_inp_batch[:,1:], 
                                                             model_read.outputs: read_out_batch[:, :-1], model_read.targets: read_out_batch[:, 1:]})   

                        #print("Time it took to run one batch for reading: ", time()-t)
                        t = time()
                        read_loss.append(batch_loss)

                        read_epoch_loss += batch_loss
                        #print(read_inp_batch.dtype, batch_logits.dtype, read_out_batch[:,1:].dtype, len(dict_char2num_x))
                        #read_old_accs[k], read_token_accs[k] , read_word_accs[k] = utils.accuracy(r_batch_logits, read_out_batch[:,1:], dict_char2num_x)
                        #print("Time it took compute analysis: ", time()-t+tt)

                        
                        read_old_accs[k], fullPred, fullTarg = utils.accuracy_prepare(r_batch_logits, read_out_batch[:,1:], dict_char2num_x)
                    
                        dists, read_token_accs[k] = sess.run([acc_object.dists, acc_object.token_acc], 
                            feed_dict={acc_object.fullPred:fullPred, acc_object.fullTarg: fullTarg})

                        read_word_accs[k] = np.count_nonzero(dists==0) / len(dists) 


            elif regime == 'lds':
                

                for k, (write_inp_batch, write_out_batch, write_alt_targs) in enumerate(utils.batch_data(X_train, Y_train, args.batch_size, Y_alt_train)):


                    _, batch_loss, write_new_targs, rat_lds, rat_corr, batch_loss_reg, w_batch_logits = sess.run([model_write.lds_optimizer, model_write.loss_lds, 
                        model_write.read_inps, model_write.rat_lds, model_write.rat_corr, model_write.loss_reg, model_write.logits], 
                                                                        feed_dict = {model_write.keep_prob:1.0, model_write.inputs: write_inp_batch[:,1:], 
                                                                            model_write.outputs: write_out_batch[:, :-1], model_write.targets: write_out_batch[:, 1:],
                                                                            model_write.alternative_targets: write_alt_targs[:,1:,:]})

                    rats_lds.append(rat_lds)
                    rats_corr.append(rat_corr)
                    lds_loss.append(batch_loss)
                    reg_loss.append(batch_loss_reg)

                    if epoch > theta_min and epoch < theta_max:
                        utils.num_to_str(write_inp_batch,w_batch_logits,write_out_batch,write_alt_targs,dict_num2char_x,dict_num2char_y)


                    write_epoch_loss += batch_loss
                    #write_old_accs[k], write_token_accs[k] , write_word_accs[k] = utils.accuracy(w_batch_logits, write_new_targs, dict_char2num_y)
                    write_old_accs[k], fullPred, fullTarg = utils.accuracy_prepare(w_batch_logits, write_out_batch[:,1:], dict_char2num_y)
                    
                    dists, write_token_accs[k] = sess.run([acc_object.dists, acc_object.token_acc], 
                            feed_dict={acc_object.fullPred:fullPred, acc_object.fullTarg: fullTarg})

                    write_word_accs[k] = np.count_nonzero(dists==0) / len(dists) 

                    # Test reading
                    if args.reading:
                        read_inp_batch = write_new_targs
                        read_out_batch = write_inp_batch
                        #read_out_batch = np.concatenate([np.ones([args.batch_size,1],dtype=np.int64) * dict_char2num_x['<GO>'], write_inp_batch],axis=1)

                        _, batch_loss, r_batch_logits = sess.run([model_read.optimizer, model_read.loss, model_read.logits], feed_dict =
                                                             {model_read.keep_prob:1.0, model_read.inputs: read_inp_batch, 
                                                             model_read.outputs: read_out_batch[:, :-1], model_read.targets: read_out_batch[:, 1:]})   
                        read_epoch_loss += batch_loss
                        #print(read_inp_batch.dtype, batch_logits.dtype, read_out_batch[:,1:].dtype, len(dict_char2num_x))
                        #t=time()
                        #read_old_accs[k], read_token_accs[k] , read_word_accs[k] = utils.accuracy(r_batch_logits, read_out_batch[:,1:], dict_char2num_x)
                        #print("Time it took compute analysis: ", time()-t+tt)
                        read_loss.append(batch_loss)

                        read_old_accs[k], fullPred, fullTarg = utils.accuracy_prepare(r_batch_logits, read_out_batch[:,1:], dict_char2num_x)
                    
                        dists, read_token_accs[k] = sess.run([acc_object.dists, acc_object.token_acc], 
                            feed_dict={acc_object.fullPred:fullPred, acc_object.fullTarg: fullTarg})

                        read_word_accs[k] = np.count_nonzero(dists==0) / len(dists) 

                
            lds_losses[epoch] = sum(lds_loss)/len(lds_loss)
            reg_losses[epoch] = sum(reg_loss)/len(reg_loss)
            lds_ratios[epoch] = sum(rats_lds)/len(rats_lds)
            corr_ratios[epoch] = sum(rats_corr)/len(rats_corr)
            read_losses[epoch] = sum(read_loss)/len(read_loss)

            print("Displayed run - Ratio correct words: " + str(corr_ratios[epoch])+" and in LdS sense: " + str(lds_ratios[epoch]))
            print("Displayed run - LdS loss is " + str(lds_losses[epoch]) + " while regular loss is" + str(reg_losses[epoch]))



            if epoch % args.save_model == 0:
                np.savez(save_path + '/write_step' + str(epoch)+'.npz', logits=w_batch_logits, dict=dict_char2num_y, targets=write_out_batch[:,1:])
                np.savez(save_path + '/read_step' + str(epoch)+'.npz', logits=r_batch_logits, dict=dict_char2num_x, targets=read_out_batch[:,1:])

            #print('Train',batch_logits.shape, write_out_batch[:,1:].shape)
            print('WRITING - Loss:{:>6.3f}  token acc:{:>6.3f},  word acc:{:>6.3f} old acc:{:>6.4f}'
                  .format(write_epoch_loss, np.mean(write_token_accs), np.mean(write_word_accs), np.mean(write_old_accs)))
            trainPerf[epoch//args.print_step, 0] = np.mean(write_token_accs)
            trainPerf[epoch//args.print_step, 1] = np.mean(write_word_accs)
            trainPerf[epoch//args.print_step, 2] = np.mean(write_old_accs)
            #print('NEW!!! WRITING - Loss:{:>6.3f}  token acc:{:>6.3f},  word acc:{:>6.3f} old acc:{:>6.4f}'
            #      .format(write_epoch_loss, np.mean(write_token_accs_2), np.mean(write_word_accs_2), np.mean(write_old_accs_2)))
            if args.reading:
                print('READING - Loss:{:>6.3f}  token acc:{:>6.3f},  word acc:{:>6.3f} old acc:{:>6.4f}'
                      .format(read_epoch_loss, np.mean(read_token_accs), np.mean(read_word_accs), np.mean(read_old_accs)))
                trainPerf[epoch//args.print_step, 3] = np.mean(read_token_accs)
                trainPerf[epoch//args.print_step, 4] = np.mean(read_word_accs)
                trainPerf[epoch//args.print_step, 5] = np.mean(read_old_accs)


            #print("Time that epoch took: ", time()-t)


            """
            # TESTING NOT PROPERLY (Feeding target as output)
            wordAccs  = np.zeros(len(X_test) // args.batch_size)
            tokenAccs = np.zeros(len(X_test) // args.batch_size)
            oldAccs   = np.zeros(len(X_test) // args.batch_size)
            
            for batch_i, (write_inp_batch, write_out_batch) in enumerate(utils.batch_data(X_test, Y_test, args.batch_size)):
                                                    
                batch_logits = sess.run(model_write.logits, feed_dict = {model_write.keep_prob:1.0, model_write.inputs: write_inp_batch,
                                                                            model_write.outputs: write_out_batch[:, :-1]})
                predictions = batch_logits.argmax(-1)
                #print('Test',batch_logits.shape,write_out_batch[:,1:].shape)
                oldAccs[batch_i], tokenAccs[batch_i] , wordAccs[batch_i] = utils.accuracy(batch_logits,write_out_batch[:,1:],dict_char2num_y, mode='train')
            print('- WRITING BAD - Accuracy on test set is for tokens{:>6.3f} and for words {:>6.3f}'.format(np.mean(tokenAccs), np.mean(wordAccs)))
            """



            # --------------- SHOW TESTING PERFORMANCE -----------------------------------


            if regime == 'normal':

                write_dec_input = np.zeros((len(X_test), 1)) + dict_char2num_y['<GO>']
                # Generate character by character (for the entire batch, weirdly)
                for i in range(y_seq_length):
 
                    #print(i)
                    #write_test_logits, write_loss, = sess.run([model_write.logits,  model_write.loss_reg], 
                    #    feed_dict={model_write.keep_prob:1.0, model_write.inputs:X_test[:,1:], model_write.outputs:write_dec_input,
                    #    model_write.targets: Y_test[:, 1:],
                    #    model_write.alternative_targets: Y_alt_test[:,1:,:]})
                    write_test_logits = sess.run(model_write.logits, 
                        feed_dict={model_write.keep_prob:1.0, model_write.inputs:X_test[:,1:], model_write.outputs:write_dec_input})
                    #print("Y!")
                    write_prediction = write_test_logits[:,-1].argmax(axis=-1)
                    #print('Loop',test_logits.shape, test_logits[:,-1].shape, prediction.shape)
                    write_dec_input = np.hstack([write_dec_input, write_prediction[:,None]])
                #print(dec_input[:,1:].shape, Y_test[:,1:].shape)
                #write_oldAcc_o, write_tokenAcc_o , write_wordAcc_o = utils.accuracy(write_dec_input[:,1:], Y_test[:,1:],dict_char2num_y, mode='test')
               
                #lds_ratios_test[epoch//args.print_step] = rat_lds
                #corr_ratios_test[epoch//args.print_step] = rat_corr
                #lds_losses_test[epoch//args.print_step] = write_loss_lds
                #reg_losses_test[epoch//args.print_step] = write_loss


                write_oldAcc, fullPred, fullTarg = utils.accuracy_prepare(write_dec_input[:,1:], Y_test[:,1:],dict_char2num_y, mode='test')
                dists, write_tokenAcc = sess.run([acc_object.dists, acc_object.token_acc], 
                        feed_dict={acc_object.fullPred:fullPred, acc_object.fullTarg: fullTarg})
                write_wordAcc  = np.count_nonzero(dists==0) / len(dists) 

                print('WRITING - Accuracy on test set is for tokens{:>6.3f} and for words {:>6.3f}'.format(write_tokenAcc, write_wordAcc))
                #print('WRITING - Accuracy on test set is for tokens{:>6.3f} and for words {:>6.3f}'.format(write_tokenAcc_o, write_wordAcc_o))

                testPerf[epoch//args.print_step, 0] = write_tokenAcc
                testPerf[epoch//args.print_step, 1] = write_wordAcc

                # Test READING
                if args.reading:
                    read_dec_input = np.zeros((len(X_test), 1)) + dict_char2num_x['<GO>']
                    # Generate character by character (for the entire batch, weirdly)
                    for i in range(x_seq_length):
                        #print(i)
                        read_test_logits = sess.run(model_read.logits, 
                            feed_dict={model_read.keep_prob:1.0, model_read.inputs:Y_test[:,1:], model_read.outputs:read_dec_input})
                        read_prediction = read_test_logits[:,-1].argmax(axis=-1)
                        #print("W")
                        #print('Loop',test_logits.shape, test_logits[:,-1].shape, prediction.shape)
                        read_dec_input = np.hstack([read_dec_input, read_prediction[:,None]])
                    #print(dec_input[:,1:].shape, Y_test[:,1:].shape)
                    #read_oldAc_o, read_tokenAcc_o , read_wordAcc_o = utils.accuracy(read_dec_input[:,1:], X_test[:,1:],dict_char2num_x, mode='test')

                    read_oldAcc, fullPred, fullTarg = utils.accuracy_prepare(read_dec_input[:,1:], X_test[:,1:],dict_char2num_x, mode='test')
                    dists, read_tokenAcc = sess.run([acc_object.dists, acc_object.token_acc], 
                            feed_dict={acc_object.fullPred:fullPred, acc_object.fullTarg: fullTarg})
                    read_wordAcc  = np.count_nonzero(dists==0) / len(dists) 

                    print('READING - Accuracy on test set is for tokens{:>6.3f} and for words {:>6.3f}'.format(read_tokenAcc, read_wordAcc))
                    #print('OLD - READING - Accuracy on test set is for tokens{:>6.3f} and for words {:>6.3f}'.format(read_tokenAcc_o, read_wordAcc_o))

                    testPerf[epoch//args.print_step, 2] = read_tokenAcc
                    testPerf[epoch//args.print_step, 3] = read_wordAcc

                    #read_losses[epoch//args.print_step] = read_test_loss

            elif regime == 'lds':

                write_dec_input = np.zeros((len(X_test), 1)) + dict_char2num_y['<GO>']
                # Generate character by character (for the entire batch, weirdly)
                for i in range(y_seq_length):
                    """
                    write_test_logits, write_loss_lds, write_loss, rat_lds, rat_corr = sess.run([model_write.logits,
                        model_write.loss_lds, model_write.loss_reg, model_write.rat_lds, model_write.rat_corr],
                        feed_dict={model_write.keep_prob:1.0, model_write.inputs:X_test[:,1:], model_write.outputs:write_dec_input,
                        model_write.targets: Y_test[:, 1:],
                        model_write.alternative_targets: Y_alt_test[:,1:,:]})
                    """

                    write_test_logits = sess.run(model_write.logits, 
                        feed_dict={model_write.keep_prob:1.0, model_write.inputs:X_test[:,1:], model_write.outputs:write_dec_input})
                    write_prediction = write_test_logits[:,-1].argmax(axis=-1)
                    #print('Loop',test_logits.shape, test_logits[:,-1].shape, prediction.shape)
                    write_dec_input = np.hstack([write_dec_input, write_prediction[:,None]])

                # Now the generated sequence need to be compared with the alternative targets:
                write_test_new_targs = utils.lds_compare(write_dec_input[:,1:],Y_test[:,1:], Y_alt_test[:,1:])


                #lds_ratios_test[epoch//args.print_step] = rat_lds
                #corr_ratios_test[epoch//args.print_step] = rat_corr
                #lds_losses_test[epoch//args.print_step] = write_loss_lds
                #reg_losses_test[epoch//args.print_step] = write_loss



                #write_oldAcc_o, write_tokenAcc_o , write_wordAcc_o = utils.accuracy(write_dec_input, write_test_new_targs,dict_char2num_y, mode='test')

                write_oldAcc, fullPred, fullTarg = utils.accuracy_prepare(write_dec_input, write_test_new_targs,dict_char2num_y, mode='test')
                dists, write_tokenAcc = sess.run([acc_object.dists, acc_object.token_acc], 
                        feed_dict={acc_object.fullPred:fullPred, acc_object.fullTarg: fullTarg})
                write_wordAcc  = np.count_nonzero(dists==0) / len(dists) 

                print('WRITING - Accuracy on test set is for tokens{:>6.3f} and for words {:>6.3f}'.format(write_tokenAcc, write_wordAcc))
                #print('OLD WRITING - Accuracy on test set is for tokens{:>6.3f} and for words {:>6.3f}'.format(write_tokenAcc_o, write_wordAcc_o))

                testPerf[epoch//args.print_step, 0] = write_tokenAcc
                testPerf[epoch//args.print_step, 1] = write_wordAcc

                # Test READING
                if args.reading:
                    read_dec_input = np.zeros((len(X_test), 1)) + dict_char2num_x['<GO>']
                    read_test_new_inp = write_test_new_targs
                    # Generate character by character (for the entire batch, weirdly)
                    for i in range(x_seq_length):
                        read_test_logits = sess.run(model_read.logits, feed_dict={model_read.keep_prob:1.0, 
                                model_read.inputs:read_test_new_inp, model_read.outputs:read_dec_input})
                        read_prediction = read_test_logits[:,-1].argmax(axis=-1)
                        #print('Loop',test_logits.shape, test_logits[:,-1].shape, prediction.shape)
                        read_dec_input = np.hstack([read_dec_input, read_prediction[:,None]])
                    #print(dec_input[:,1:].shape, Y_test[:,1:].shape)
                    #read_oldAcc_o, read_tokenAcc_o , read_wordAcc_o = utils.accuracy(read_dec_input[:,1:], X_test[:,1:],dict_char2num_x, mode='test')

                    read_oldAcc, fullPred, fullTarg = utils.accuracy_prepare(read_dec_input[:,1:], X_test[:,1:],dict_char2num_x, mode='test')
                    dists, read_tokenAcc = sess.run([acc_object.dists, acc_object.token_acc], 
                            feed_dict={acc_object.fullPred:fullPred, acc_object.fullTarg: fullTarg})
                    read_wordAcc  = np.count_nonzero(dists==0) / len(dists) 
                    #read_losses[epoch//args.print_step] = read_test_loss

                    print('READING - Accuracy on test set is for tokens{:>6.3f} and for words {:>6.3f}'.format(read_tokenAcc, read_wordAcc))
                    #print('OLD READING - Accuracy on test set is for tokens{:>6.3f} and for words {:>6.3f}'.format(read_tokenAcc_o, read_wordAcc_o))
                    testPerf[epoch//args.print_step, 2] = read_tokenAcc
                    testPerf[epoch//args.print_step, 3] = read_wordAcc
            

        if epoch % args.save_model == 0:
            saver_write.save(sess, save_path + '/Model_write', global_step=epoch, write_meta_graph=False)
            if args.reading:
                saver_read.save(sess, save_path + '/Model_read', global_step=epoch, write_meta_graph=False)
                np.savez(save_path + '/metrics.npz', trainPerf=trainPerf, testPerf=testPerf, lds_ratio=lds_ratios,lds_loss=lds_losses, 
                    reg_loss=reg_losses,corr_ratio=corr_ratios, read_losses=read_losses)
        if args.epochs // 2 == epoch and regime == 'lds':
            regime = 'normal'
            model_write.learn_type = 'normal'
            print("Training regime changed to normal")


                    
    saver_write.save(sess, save_path + '/Model_write', global_step=epoch, write_meta_graph=False)
    if args.reading:
        saver_read.save(sess, save_path + '/Model_read', global_step=epoch, write_meta_graph=False)           
                

    print(" Training done, model_write saved in file: %s" % save_path + ' ' + os.path.abspath(save_path))

    #np.savetxt(save_path+'/train.txt', trainPerf, delimiter=',')   
    #np.savetxt(save_path+'/test.txt', testPerf, delimiter=',')  
    np.savez(save_path + '/metrics.npz', trainPerf=trainPerf, testPerf=testPerf, lds_ratio=lds_ratios,lds_loss=lds_losses, 
        reg_loss=reg_losses,corr_ratio=corr_ratios)

    if args.show_plot:
        ax = plt.subplot(111) 
        plt.plot(np.linspace(1,args.epochs, trainPerf.shape[0]), trainPerf[:,0], label="trainToken")
        plt.plot(np.linspace(1,args.epochs, trainPerf.shape[0]), trainPerf[:,1], label="trainWord")
        plt.plot(np.linspace(1,args.epochs, trainPerf.shape[0]), testPerf[:,0], label="testToken")
        plt.plot(np.linspace(1,args.epochs, trainPerf.shape[0]), testPerf[:,1], label="testWord")
        ax.legend(loc='best')
        plt.show()




    # TESTING

    # Set initial decoder input to be 0
    dec_input = np.zeros((len(X_test), 1)) + dict_char2num_y['<GO>']

    # Generate character by character (for the entire batch, weirdly)
    for i in range(y_seq_length):
        test_logits = sess.run(model_write.logits, feed_dict={model_write.keep_prob:1.0, model_write.inputs:X_test[:,1:], model_write.outputs:dec_input})
        prediction = test_logits[:,-1].argmax(axis=-1)
        dec_input = np.hstack([dec_input, prediction[:,None]])

    oldAcc, tokenAcc , wordAcc = utils.accuracy(dec_input[:,1:], Y_test[:,1:], dict_char2num_y, mode='test')

    print('Accuracy on test set is for tokens{:>6.3f} and for words {:>6.3f}'.format(tokenAcc, wordAcc))

    print("DONE!")   

    # Call prediction class here
    

    '''# Show a prediction of the model_write
                print()
                print("Some examples generated by the model")
                num_preds = 2
                for k in range(num_preds):
            
                    spoken_word = X_train[np.random.randint(len(X_train))]
                    utils.write_word(spoken_word)
            
                    if args.reading:
                        written_word = Y_train[np.random.randint(len(Y_train))]
                        utils.read_word(written_word)'''











"""
    num_preds = 2
    source_chars = [[dict_num2char_x[x_index] for x_index in sent if dict_num2char_x[x_index]!="<PAD>"] for sent in write_inp_batch[:num_preds]]
    dest_chars = [[dict_num2char_y[y_index] for y_index in sent] for sent in dec_input[:num_preds, 1:]]

    for date_in, date_out in zip(source_chars, dest_chars):
        print(''.join(date_in)+' => '+''.join(date_out))"""









############ CODE VON JITHENDAR ###########





'''

        
        # variables to get the training error in current epoch
        train_loss, train_batches = 0., 0
        train_opt, train_load = 0., 0.
        train_start = time.time()
        print('Starting to train in epoch {}'.format(epoch + 1))
        for data in tqdm(dataset.flow_merge('train', batch_size=args.batch_size, phased=False),
                         total=num_batches):
            current_labels = data[1]
            sparse_labels = tf_utils.sparse_tuple_from(current_labels)
            feed_dict = {seq_len_tensor: data[0][1], target_tensor: sparse_labels}
            feed_dict[input_tensor] = data[0][0]

            opt_start = time.time()
            train_load += opt_start - batch_start
            _, error, summary = sess.run([train_step_tensor, loss_tensor,
                                          merged], feed_dict=feed_dict)

            train_writer.add_summary(summary, num_train_steps)
            train_loss += error
            train_batches += 1
            num_train_steps += 1
            batch_start = time.time()
            train_opt += batch_start - opt_start

        train_end = time.time() - train_start

        val_loss, val_batches = 0., 0
        val_start = time.time()
        val_ler = 0.

        num_batches = dataset.num_batches('test', batch_size=args.batch_size)
        print('Starting to process {} test batches '
              'in epoch {}'.format(num_batches, epoch + 1))
        for data in tqdm(dataset.flow_merge('test', batch_size=args.batch_size, phased=False),
                         total=num_batches):
            current_labels = data[1]
            sparse_labels = tf_utils.sparse_tuple_from(current_labels)
            feed_dict = {seq_len_tensor: data[0][1], target_tensor: sparse_labels}
            feed_dict[input_tensor] = data[0][0]

            c_ler, error, summary = sess.run([ler_tensor, loss_tensor,
                                              merged], feed_dict=feed_dict)
            test_writer.add_summary(summary, num_val_steps)

            val_loss += error
            val_ler += c_ler
            val_batches += 1
            num_val_steps += 1

        val_end = time.time() - val_start
        exp.add_metric_row({'train_loss': train_loss / train_batches,
                            'test_loss': val_loss / val_batches,
                            'test_LER': val_ler / val_batches})

        print('Epoch {}, Train Loss {:.2f}, Train Time {:.2f}, '
              'Train Opt {:.2f}, Train load {:.2f}, '
              'Test Loss {:.2f}, LER {:.2f}, '
              'Test Time {:.2f}.'.format(epoch + 1, train_loss / train_batches,
                                train_end, train_opt, train_load,
                                val_loss / val_batches,
                                val_ler / val_batches, val_end))
        saver.save(sess, save_path + '/recent')
    print('Done')





'''