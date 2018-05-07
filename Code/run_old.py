    
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

# Import my files
import utils
from bLSTM import bLSTM
from eval_model import evaluation


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
    parser.add_argument('--task', default='celex', type=str,
                        help="Sets the task to solve, default is 'TIMIT_P2G', alternatives are 'Dates', 'TIMIT_G2P' "
                        " and later on more...")
    parser.add_argument('--learn_type', default='normal', type=str,
                        help="Determines the training regime. Choose from set {'normal', 'lds'}.")
    parser.add_argument('--reading', default=True, type=bool,
                        help="Specifies whether reading task is also accomplished. Default is False. ")


    # Training and recording hyperparameter
    parser.add_argument('--epochs', default=100, type=int,
                        help='The number of epochs to train on')
    parser.add_argument('--print_step', default=10, type=int,
                        help='Record training & test accuracy after every n epochs')
    parser.add_argument('--batch_size', default=160, type=int,
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
    parser.add_argument('--dropout', default=0.4, type=float,
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

    elif args.task == 'celex':
        ((inputs, targets) , (dict_char2num_x, dict_char2num_y)) = utils.celex_retrieve()

    elif args.task == 'bas':
        ((inputs, targets) , (dict_char2num_x, dict_char2num_y)) = utils.BAS_P2G_retrieve()


    # Set remaining parameter based on the processed data
    x_dict_size, num_classes, x_seq_length, y_seq_length, dict_num2char_x, dict_num2char_y = utils.set_model_params(inputs, targets, dict_char2num_x, dict_char2num_y)

    # Split data into training and testing
    indices = range(len(inputs))
    X_train, X_test,Y_train, Y_test, indices_train, indices_test = train_test_split(inputs, targets, indices, test_size=args.test_size, random_state=args.seed)




    ############## PREPARATION FOR TRAINING ##############

    model_write = bLSTM(x_seq_length, y_seq_length, x_dict_size, num_classes, args.input_embed_size, args.output_embed_size, args.num_layers, args.num_nodes, args.batch_size,
        args.learn_type, 'write', print_ratio=args.print_ratio, optimization=args.optimization ,learning_rate=args.learning_rate, LSTM_initializer=args.LSTM_initializer, 
        momentum=args.momentum, activation_fn=args.activation_fn, bidirectional=args.bidirectional)
    model_write.forward()
    model_write.backward()


    # Should the reading module be enabled?
    if args.reading:
        model_read = bLSTM(y_seq_length, x_seq_length, num_classes, x_dict_size, args.input_embed_size, args.output_embed_size, args.num_layers, args.num_nodes,
            args.batch_size, args.learn_type, 'read',print_ratio=args.print_ratio, optimization=args.optimization ,learning_rate=args.learning_rate, 
            LSTM_initializer=args.LSTM_initializer, momentum=args.momentum, activation_fn=args.activation_fn, bidirectional=args.bidirectional)
        model_read.forward()
        model_read.backward()

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
    tf.summary.scalar('Seq2Seq Loss', model_write.loss)

    # builds the histogram of GRU activations
    tf.summary.histogram('GRU activations', model_write.all_logits)

    # calculates the total number of parameters in the network
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    print('{} trainable parameters in the network.'.format(total_parameters))

    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(save_path + '/train', sess.graph)
    test_writer = tf.summary.FileWriter(save_path + '/test')



    # defining the saver object
    saver = tf.train.Saver(max_to_keep=4)

    if args.restore:
        utils.retrieve_model()
    else:
        # tensor to initialize the variables
        init_tensor = tf.global_variables_initializer()
        # initializing the variables
        sess.run(init_tensor)

    #if args.restore:
    #saver.restore(sess, _model_writePath) #Yes, no need to add ".index"

    num_train_steps, num_val_steps = 0, 0

    if args.reading:
        trainPerf = np.zeros([args.epochs//args.print_step + 1, 6])
        testPerf = np.zeros([args.epochs//args.print_step + 1, 6])
    else:
        trainPerf = np.zeros([args.epochs//args.print_step + 1, 3])
        testPerf = np.zeros([args.epochs//args.print_step + 1, 3])

    print('\n Starting training \n ')
    for epoch in range(args.epochs):

        print('Epoch ', epoch + 1)
            
        # Regular training (do not show performance)
        if epoch % args.print_step != 0 :

            # Modify this if learn_type is lds: foor loop needs to incorporate alternative targets.
            
            for batch_i, (write_inp_batch, write_out_batch) in enumerate(utils.batch_data(X_train, Y_train, args.batch_size)):
                
                # Train Writing

                if args.learn_type == 'normal':
                    _, batch_loss, batch_logits = sess.run([model_write.optimizer, model_write.loss, model_write.logits], feed_dict = 
                                                            {model_write.keep_prob: args.dropout, model_write.inputs: write_inp_batch, 
                                                            model_write.outputs: write_out_batch[:, :-1], model_write.targets: write_out_batch[:, 1:]})
                elif args.learn_type == 'lds':
                    _, batch_loss, batch_logits = sess.run([model_write.optimizer, model_write.loss, model_write.logits], feed_dict = 
                                                            {model_write.keep_prob: args.dropout, model_write.inputs: write_inp_batch, 
                                                            model_write.outputs: write_out_batch[:, :-1], model_write.targets: write_out_batch[:, 1:], 
                                                            model_write.alternative_targets: alternative_targets})
                
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


                # Train READING:
                if args.reading:

                    if args.learn_type == 'normal':
                        read_inp_batch = write_out_batch[:,1:]
                        read_out_batch = np.concatenate([np.ones([args.batch_size,1],dtype=np.int64) * dict_char2num_x['<GO>'], write_inp_batch],axis=1)

                        _, batch_loss, batch_logits = sess.run([model_read.optimizer, model_read.loss, model_read.logits], feed_dict = 
                                                        {model_read.keep_prob:args.dropout, model_read.inputs: read_inp_batch, 
                                                        model_read.outputs:read_out_batch[:,:-1], model_read.targets:read_out_batch[:,1:]})

                    #elif args.learn_type == 'lds':



                
        else: # Display performance
            
            # Allocate variables
            write_word_accs = np.zeros(len(X_train)// args.batch_size)
            write_token_accs = np.zeros(len(X_train)// args.batch_size)
            write_old_accs = np.zeros(len(X_train)// args.batch_size)
            write_epoch_loss = 0

            read_word_accs = np.zeros(len(X_train)// args.batch_size)
            read_token_accs = np.zeros(len(X_train)// args.batch_size)
            read_old_accs = np.zeros(len(X_train)// args.batch_size)
            read_epoch_loss = 0

            for k, (write_inp_batch, write_out_batch) in enumerate(utils.batch_data(X_train, Y_train, args.batch_size)):
                
                # Test writing
                _, batch_loss, batch_logits = sess.run([model_write.optimizer, model_write.loss, model_write.logits], feed_dict =
                                                         {model_write.keep_prob:1.0, model_write.inputs: write_inp_batch, 
                                                         model_write.outputs: write_out_batch[:, :-1], model_write.targets: write_out_batch[:, 1:]})   
                write_epoch_loss += batch_loss

                write_old_accs[k], write_token_accs[k] , write_word_accs[k] = utils.accuracy(batch_logits, write_out_batch[:,1:], dict_char2num_y)

                # Test reading
                if args.reading:
                    read_inp_batch = write_out_batch[:,1:]
                    read_out_batch = np.concatenate([np.ones([args.batch_size,1],dtype=np.int64) * dict_char2num_x['<GO>'], write_inp_batch],axis=1)

                    _, batch_loss, batch_logits = sess.run([model_read.optimizer, model_read.loss, model_read.logits], feed_dict =
                                                         {model_read.keep_prob:1.0, model_read.inputs: read_inp_batch, 
                                                         model_read.outputs: read_out_batch[:, :-1], model_read.targets: read_out_batch[:, 1:]})   
                    read_epoch_loss += batch_loss
                    #print(read_inp_batch.dtype, batch_logits.dtype, read_out_batch[:,1:].dtype, len(dict_char2num_x))
                    read_old_accs[k], read_token_accs[k] , read_word_accs[k] = utils.accuracy(batch_logits, read_out_batch[:,1:], dict_char2num_x)




                

            if epoch == 0 or epoch == 200:
                np.savez('step' + str(epoch)+'.npz', logits=batch_logits, dict=dict_char2num_y, targets=write_out_batch[:,1:])

            #print('Train',batch_logits.shape, write_out_batch[:,1:].shape)
            print('WRITING - Loss:{:>6.3f}  token acc:{:>6.3f},  word acc:{:>6.3f} old acc:{:>6.4f}'
                  .format(write_epoch_loss, np.mean(write_token_accs), np.mean(write_word_accs), np.mean(write_old_accs)))
            trainPerf[epoch//args.print_step, 0] = np.mean(write_token_accs)
            trainPerf[epoch//args.print_step, 1] = np.mean(write_word_accs)
            trainPerf[epoch//args.print_step, 2] = np.mean(write_old_accs)

            if args.reading:
                print('READING - Loss:{:>6.3f}  token acc:{:>6.3f},  word acc:{:>6.3f} old acc:{:>6.4f}'
                      .format(read_epoch_loss, np.mean(read_token_accs), np.mean(read_word_accs), np.mean(read_old_accs)))
                trainPerf[epoch//args.print_step, 3] = np.mean(read_token_accs)
                trainPerf[epoch//args.print_step, 4] = np.mean(read_word_accs)
                trainPerf[epoch//args.print_step, 5] = np.mean(read_old_accs)

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


            # TESTING 2
            # Set initial decoder input to be 0

            write_dec_input = np.zeros((len(X_test), 1)) + dict_char2num_y['<GO>']
            # Generate character by character (for the entire batch, weirdly)
            for i in range(y_seq_length):
                write_test_logits = sess.run(model_write.logits, feed_dict={model_write.keep_prob:1.0, model_write.inputs:X_test, model_write.outputs:write_dec_input})
                write_prediction = write_test_logits[:,-1].argmax(axis=-1)
                #print('Loop',test_logits.shape, test_logits[:,-1].shape, prediction.shape)
                write_dec_input = np.hstack([write_dec_input, write_prediction[:,None]])
            #print(dec_input[:,1:].shape, Y_test[:,1:].shape)
            write_oldAcc, write_tokenAcc , write_wordAcc = utils.accuracy(write_dec_input[:,1:], Y_test[:,1:],dict_char2num_y, mode='test')
            print('WRITING - Accuracy on test set is for tokens{:>6.3f} and for words {:>6.3f}'.format(write_tokenAcc, write_wordAcc))
            testPerf[epoch//args.print_step, 0] = write_tokenAcc
            testPerf[epoch//args.print_step, 1] = write_wordAcc

            # Test READING
            if args.reading:
                read_dec_input = np.zeros((len(X_test), 1)) + dict_char2num_x['<GO>']
                # Generate character by character (for the entire batch, weirdly)
                for i in range(x_seq_length):
                    read_test_logits = sess.run(model_read.logits, feed_dict={model_read.keep_prob:1.0, model_read.inputs:Y_test[:,1:], model_read.outputs:read_dec_input})
                    read_prediction = read_test_logits[:,-1].argmax(axis=-1)
                    #print('Loop',test_logits.shape, test_logits[:,-1].shape, prediction.shape)
                    read_dec_input = np.hstack([read_dec_input, read_prediction[:,None]])
                #print(dec_input[:,1:].shape, Y_test[:,1:].shape)
                read_oldAcc, read_tokenAcc , read_wordAcc = utils.accuracy(read_dec_input[:,1:], X_test[:,1:],dict_char2num_x, mode='test')
                print('READING - Accuracy on test set is for tokens{:>6.3f} and for words {:>6.3f}'.format(read_tokenAcc, read_wordAcc))
                testPerf[epoch//args.print_step, 0] = read_tokenAcc
                testPerf[epoch//args.print_step, 1] = read_wordAcc
            

        if epoch % args.save_model == 0:
            saver.save(sess, save_path + '/Model', global_step=epoch, write_meta_graph=False)

                    
               
                
    saver.save(sess, save_path + '/MODEL',write_meta_graph=False)

    print(" Training done, model_write saved in file: %s" % save_path + ' ' + os.path.abspath(save_path))

    np.savetxt(save_path+'/train.txt', trainPerf, delimiter=',')   
    np.savetxt(save_path+'/test.txt', testPerf, delimiter=',')  

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
        test_logits = sess.run(model_write.logits, feed_dict={model_write.keep_prob:1.0, model_write.inputs:X_test, model_write.outputs:dec_input})
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