import warnings
import sys
import os
import argparse
import numpy as np
import tensorflow as tf

# Import functions from some modules
from sklearn.model_selection import train_test_split
from test_tube import Experiment
from time import time

# Import my files
from invspell.utils import acc_new
from invspell import utils
from invspell.bLSTM import bLSTM

#from eval_model import evaluation
warnings.filterwarnings("ignore",category=FutureWarning)


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
    parser.add_argument('--data_dir', default='data', type=str,
                        help='Directory to read the data from')

    # Task hyperparameter
    parser.add_argument('--task', default='fibel', type=str,
                        help="Sets the task to solve, default is 'TIMIT_P2G', alternatives are 'Dates', 'TIMIT_G2P' "
                        " and later on more...")
    parser.add_argument('--learn_type', default='normal', type=str,
                        help="Determines the training regime. Choose from set {'normal', 'lds', 'interleaved'}.")
    parser.add_argument('--reading', default=False, type=bool,
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
    parser.add_argument('--input_embed_size', default=96, type=int,
                        help='The feature space dimensionality for the input characters')
    parser.add_argument('--output_embed_size', default=96, type=int,
                        help='The feature space dimensionality for the output characters')
    parser.add_argument('--num_nodes', default=128, type=int,
                        help='The dimensionality of the LSTM cell')
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

    print("READING IS ", args.reading)

##########################    PROCESS ARGUMENTS     ####################################

    if args.log_dir == 0:
        log_dir = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), 'logs')
    elif args.log_dir == 1:
        log_dir = "../../"

    print('\n You have choosen the following options: \n ', args, '\n')
    file_name = args.file_name + str(args.run_id)
    save_path = os.path.join(log_dir,'Models', args.task, args.learn_type + '_' + file_name)
    test_tube = save_path
    data_dir = args.data_dir

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
        #gpu_options = tf.GPUOptions(allow_growth=True)
        #sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.99
        print(config.gpu_options.per_process_gpu_memory_fraction)
        sess = tf.Session(config = config)


    # setting a random seed for reproducibility
    np.random.seed(args.seed)



    # LOAD DATA
    if args.task == 'celex' :
        ((inputs, targets) , (dict_char2num_x, dict_char2num_y), alt_targets) = utils.celex_retrieve(data_dir)
        mas = 96

    elif args.task == 'celex_all':
        ((inputs, targets) , (dict_char2num_x, dict_char2num_y), alt_targets) = utils.celex_all_retrieve(data_dir)
        mas = 100

    elif args.task == 'childlex':
        ((inputs, targets) , (dict_char2num_x, dict_char2num_y), alt_targets) = utils.childlex_retrieve(data_dir)
        mas = 100

    elif args.task == 'childlex_all':
        ((inputs, targets) , (dict_char2num_x, dict_char2num_y), alt_targets) = utils.childlex_all_retrieve(data_dir)
        mas = 43200

    elif args.task == 'fibel':
        ((inputs, targets) , (dict_char2num_x, dict_char2num_y), alt_targets) = utils.fibel_retrieve(data_dir)
        lektions_inds = [9,14,20,28,36,46,58,77,99,121,154,174]
        mas = 810




    # -------------------------------------------- REGULAR TRAINING SETUP --------------------------------------------------- #

    # Set remaining parameter based on the processed data
    x_dict_size, num_classes, x_seq_length, y_seq_length, dict_num2char_x, dict_num2char_y = utils.set_model_params(inputs, targets, dict_char2num_x, dict_char2num_y)


    # Split data into training and testing
    indices = range(len(inputs))
    

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

    if args.learn_type == 'lds' or args.learn_type == 'intervened' or args.learn_type == 'intervened + interleaved' or args.learn_type == 'interleaved':
        regime = 'lds'
    elif args.learn_type == 'normal' :
        regime = 'normal'
    else:
        raise ValueError('Wrong learning type given')


    print("REGIME IS ", regime)




    #tf.reset_default_graph()
    with tf.variable_scope('writing'):
        model_write = bLSTM(x_seq_length, y_seq_length, x_dict_size, num_classes, args.input_embed_size, args.output_embed_size, args.num_layers, args.num_nodes, args.batch_size,
            args.learn_type, 'write', mas, print_ratio=args.print_ratio, optimization=args.optimization ,learning_rate=args.learning_rate, LSTM_initializer=args.LSTM_initializer, 
            momentum=args.momentum, activation_fn=args.activation_fn, bidirectional=args.bidirectional)
        model_write.forward()
        model_write.backward()



    # Should the reading module be enabled?
    if args.reading:
        with tf.variable_scope('reading'):
            model_read = bLSTM(y_seq_length, x_seq_length, num_classes, x_dict_size, args.input_embed_size, args.output_embed_size, args.num_layers, args.num_nodes,
                args.batch_size, 'normal', 'read', mas, print_ratio=args.print_ratio, optimization=args.optimization ,learning_rate=args.learning_rate, 
                LSTM_initializer=args.LSTM_initializer, momentum=args.momentum, activation_fn=args.activation_fn, bidirectional=args.bidirectional)
            model_read.forward()
            model_read.backward()


    exp = Experiment(name='', save_dir=test_tube)
    # First K arguments are in the same order like the ones to initialize the bLSTM, this simplifies restoring
    
    exp.tag({'inp_len':x_seq_length, 'out_len':y_seq_length, 'x_dict_size':x_dict_size, 'num_classes':num_classes, 'input_embed':args.input_embed_size,
                        'output_embed':args.output_embed_size, 'num_layers':args.num_layers, 'nodes/Layer':args.num_nodes, 'batch_size':args.batch_size, 'learn_type':
                        args.learn_type, 'task': 'write', 'print_ratio':args.print_ratio, 'optimization':str(args.optimization), 'lr': args.learning_rate,
                        'LSTM_initializer':str(args.LSTM_initializer), 'momentum':args.momentum,'ActFctn':str(args.activation_fn), 'bidirectional': args.bidirectional,  
                         'Write+Read = ': args.reading, 'epochs': args.epochs,  'seed':args.seed,'restored':args.restore, 'dropout':args.dropout, 'train_indices':
                         indices_train, 'test_indices':indices_test})
    


    # Accuracy object
    acc_object  = acc_new()
    acc_object.accuracy()


    if args.restore:
        utils.retrieve_model()
    else:

        # tensor to initialize the variables
        init_tensor = tf.global_variables_initializer()
        saver = tf.train.Saver()

        # Finalize graph
        g = tf.get_default_graph()
        g.finalize()
        # initializing the variables
        sess.run(init_tensor)



    if args.reading:
        trainPerf = np.zeros([args.epochs//args.print_step + 1, 6])
        testPerf = np.zeros([args.epochs//args.print_step + 1, 6])
    else:
        trainPerf = np.zeros([args.epochs//args.print_step + 1, 3])
        testPerf = np.zeros([args.epochs//args.print_step + 1, 3])
        
    # To save the losses
    lds_losses = np.zeros((args.epochs,1))
    write_losses = np.zeros((args.epochs,1))
    read_losses = np.zeros((args.epochs,1))

    # To save ratio of LdS correct words
    lds_ratios = np.zeros((args.epochs,1))
    lds_ratios_test = np.zeros((args.epochs//args.print_step + 1,1))




    lt = []

    print('\n Starting training \n ')
    for epoch in range(args.epochs):

        print('Epoch ', epoch + 1)
        lt.append(regime)
        t = time()    


        if regime == 'normal':
        
            for k, (write_inp_batch, write_out_batch, write_alt_targs) in enumerate(utils.batch_data(X_train, Y_train, args.batch_size,Y_alt_train)):
            
            # Train Writing
                tt=time()
                _, batch_loss, w_batch_logits, loss_lds, rat_lds = sess.run([model_write.optimizer, model_write.loss, model_write.logits, 
                    model_write.loss_lds, model_write.rat_lds], feed_dict = 
                                                        {model_write.keep_prob: args.dropout, model_write.inputs: write_inp_batch[:, 1:], 
                                                        model_write.outputs: write_out_batch[:, :-1], model_write.targets: write_out_batch[:, 1:],
                                                        model_write.alternative_targets: write_alt_targs[:,1:,:]})
                #print("Time on batch of training took ", time()-tt)

                if args.reading:

                    read_inp_batch = write_out_batch
                    read_out_batch = write_inp_batch

                    _, batch_loss, batch_logits = sess.run([model_read.optimizer, model_read.loss, model_read.logits], feed_dict = 
                                                    {model_read.keep_prob:args.dropout, model_read.inputs: read_inp_batch[:,1:], 
                                                    model_read.outputs:read_out_batch[:,:-1], model_read.targets:read_out_batch[:,1:]})


        elif regime == 'lds':


            for k, (write_inp_batch, write_out_batch, write_alt_targs) in enumerate(utils.batch_data(X_train, Y_train, args.batch_size, Y_alt_train)):

                tt=time()
                _, batch_loss, write_new_targs, rat_lds, rat_corr, batch_loss_reg, w_batch_logits= sess.run([model_write.lds_optimizer, model_write.loss_lds, model_write.read_inps, 
                    model_write.rat_lds, model_write.rat_corr, model_write.loss_reg, model_write.logits], 
                                feed_dict = 
                                                        {model_write.keep_prob:args.dropout, model_write.inputs: write_inp_batch[:,1:], 
                                                        model_write.outputs: write_out_batch[:, :-1], model_write.targets: write_out_batch[:, 1:], 
                                                        model_write.alternative_targets: write_alt_targs[:,1:,:]})

                if args.reading:
                    read_inp_batch = write_new_targs
                    read_out_batch = write_inp_batch
                    _, batch_loss, batch_logits = sess.run([model_read.optimizer, model_read.loss, model_read.logits], feed_dict = 
                                                    {model_read.keep_prob:args.dropout, model_read.inputs: read_inp_batch, 
                                                    model_read.outputs:read_out_batch[:,:-1], model_read.targets:read_out_batch[:,1:]})
      
        print("The regular training took: ", time()-t)
        tt=time()
        # ---------------- SHOW TRAINING PERFORMANCE -------------------------
        
        rats_lds = []
        lds_loss = []
        write_loss = []
        read_loss = []

        # Allocate variables
        write_word_accs = np.zeros(len(X_train)// args.batch_size)
        write_token_accs = np.zeros(len(X_train)// args.batch_size)
        write_old_accs = np.zeros(len(X_train)// args.batch_size)

        read_word_accs = np.zeros(len(X_train)// args.batch_size)
        read_token_accs = np.zeros(len(X_train)// args.batch_size)
        read_old_accs = np.zeros(len(X_train)// args.batch_size)
            


        if regime == 'normal':

            for k, (write_inp_batch, write_out_batch,write_alt_targs) in enumerate(utils.batch_data(X_train, Y_train, args.batch_size, Y_alt_train)):
                batch_loss, w_batch_logits, loss_lds, rat_lds, rat_corr, x = sess.run([model_write.loss, model_write.logits, 
                    model_write.loss_lds, model_write.rat_lds, model_write.rat_corr, model_write.fc1], feed_dict =
                                                         {model_write.keep_prob:1.0, model_write.inputs: write_inp_batch[:,1:], 
                                                         model_write.outputs: write_out_batch[:, :-1], model_write.targets: write_out_batch[:, 1:],
                                                        model_write.alternative_targets: write_alt_targs[:,1:,:]})

                fullPred, fullTarg = utils.accuracy_prepare(w_batch_logits, write_out_batch[:,1:], dict_char2num_y)
                dists, write_token_accs[k] = sess.run([acc_object.dists, acc_object.token_acc], 
                        feed_dict={acc_object.fullPred:fullPred, acc_object.fullTarg: fullTarg})
                write_word_accs[k] = np.count_nonzero(dists==0) / len(dists) 


                rats_lds.append(rat_lds)
                lds_loss.append(loss_lds)
                write_loss.append(batch_loss)

                if epoch > theta_min and epoch < theta_max:
                    utils.num_to_str(write_inp_batch,w_batch_logits,write_out_batch,write_alt_targs,dict_num2char_x,dict_num2char_y)

                #print("Time it took compute analysis: ", time()-tt)

                # Test reading
                if args.reading:
                    read_inp_batch = write_out_batch
                    read_out_batch = write_inp_batch
                    #read_out_batch = np.concatenate([np.ones([args.batch_size,1],dtype=np.int64) * dict_char2num_x['<GO>'], write_inp_batch],axis=1)

                    batch_loss, r_batch_logits = sess.run([model_read.loss, model_read.logits], feed_dict =
                                                         {model_read.keep_prob:1.0, model_read.inputs: read_inp_batch[:,1:], 
                                                         model_read.outputs: read_out_batch[:, :-1], model_read.targets: read_out_batch[:, 1:]})   

                    read_loss.append(batch_loss)
                    
                    fullPred, fullTarg = utils.accuracy_prepare(r_batch_logits, read_out_batch[:,1:], dict_char2num_x)
                
                    dists, read_token_accs[k] = sess.run([acc_object.dists, acc_object.token_acc], 
                        feed_dict={acc_object.fullPred:fullPred, acc_object.fullTarg: fullTarg})

                    read_word_accs[k] = np.count_nonzero(dists==0) / len(dists) 


        elif regime == 'lds':
            
            for k, (write_inp_batch, write_out_batch, write_alt_targs) in enumerate(utils.batch_data(X_train, Y_train, args.batch_size, Y_alt_train)):


                batch_loss, write_new_targs, rat_lds, rat_corr, batch_loss_reg, w_batch_logits = sess.run([model_write.loss_lds, 
                    model_write.read_inps, model_write.rat_lds, model_write.rat_corr, model_write.loss_reg, model_write.logits], 
                                                                    feed_dict = {model_write.keep_prob:1.0, model_write.inputs: write_inp_batch[:,1:], 
                                                                        model_write.outputs: write_out_batch[:, :-1], model_write.targets: write_out_batch[:, 1:],
                                                                        model_write.alternative_targets: write_alt_targs[:,1:,:]})
                #print("Ratio of LdS correct words ", str(rat_lds))
                rats_lds.append(rat_lds)
                lds_loss.append(batch_loss)
                write_loss.append(batch_loss_reg)

                if epoch > theta_min and epoch < theta_max:
                    utils.num_to_str(write_inp_batch,w_batch_logits,write_out_batch,write_alt_targs,dict_num2char_x,dict_num2char_y)
                _ = utils.lds_compare(w_batch_logits,write_out_batch[:, 1:], write_alt_targs[:,1:,:], dict_num2char_y, 'train')

                fullPred, fullTarg = utils.accuracy_prepare(w_batch_logits, write_out_batch[:,1:], dict_char2num_y)
                
                dists, write_token_accs[k] = sess.run([acc_object.dists, acc_object.token_acc], 
                        feed_dict={acc_object.fullPred:fullPred, acc_object.fullTarg: fullTarg})

                write_word_accs[k] = np.count_nonzero(dists==0) / len(dists) 

                # Test reading
                if args.reading:
                    read_inp_batch = write_new_targs
                    read_out_batch = write_inp_batch

                    batch_loss, r_batch_logits = sess.run([model_read.loss, model_read.logits], feed_dict =
                                                         {model_read.keep_prob:1.0, model_read.inputs: read_inp_batch, 
                                                         model_read.outputs: read_out_batch[:, :-1], model_read.targets: read_out_batch[:, 1:]})   
                    read_loss.append(batch_loss)

                    fullPred, fullTarg = utils.accuracy_prepare(r_batch_logits, read_out_batch[:,1:], dict_char2num_x)
                
                    dists, read_token_accs[k] = sess.run([acc_object.dists, acc_object.token_acc], 
                        feed_dict={acc_object.fullPred:fullPred, acc_object.fullTarg: fullTarg})

                    read_word_accs[k] = np.count_nonzero(dists==0) / len(dists) 

        
        lds_losses[epoch] = sum(lds_loss)
        write_losses[epoch] = sum(write_loss)
        lds_ratios[epoch] = sum(rats_lds)/len(rats_lds)
        if args.reading:
            read_losses[epoch] = sum(read_loss)


        print("RUN - Ratio correct words: " + str(np.mean(write_word_accs))+" and in LdS sense: " + str(lds_ratios[epoch]))
        print("Displayed run - LdS loss is " + str(lds_losses[epoch]) + " while regular loss is" + str(write_losses[epoch]))

        if epoch % args.save_model == 0 and epoch > 1:
            np.savez(save_path + '/write_step' + str(epoch)+'.npz', logits=w_batch_logits, dict=dict_char2num_y, targets=write_out_batch[:,1:])
            np.savez(save_path + '/read_step' + str(epoch)+'.npz', logits=r_batch_logits, dict=dict_char2num_x, targets=read_out_batch[:,1:])

        print('WRITING - Loss:{:>6.3f}  token acc:{:>6.3f},  word acc:{:>6.3f} old acc:{:>6.4f}'
              .format(np.sum(write_loss), np.mean(write_token_accs), np.mean(write_word_accs), np.mean(write_old_accs)))
        trainPerf[epoch//args.print_step, 0] = np.mean(write_token_accs)
        trainPerf[epoch//args.print_step, 1] = np.mean(write_word_accs)
        trainPerf[epoch//args.print_step, 2] = np.mean(write_old_accs)

        if args.reading:
            print('READING - Loss:{:>6.3f}  token acc:{:>6.3f},  word acc:{:>6.3f} old acc:{:>6.4f}'
                  .format(np.sum(read_loss), np.mean(read_token_accs), np.mean(read_word_accs), np.mean(read_old_accs)))
            trainPerf[epoch//args.print_step, 3] = np.mean(read_token_accs)
            trainPerf[epoch//args.print_step, 4] = np.mean(read_word_accs)
            trainPerf[epoch//args.print_step, 5] = np.mean(read_old_accs)

        print("TIME all the recording of training toook ", time()-tt)
        tt=time()
        # --------------- SHOW TESTING PERFORMANCE -----------------


        if regime == 'normal':


            write_dec_input = np.zeros((len(X_test), 1)) + dict_char2num_y['<GO>']
            # Generate character by character (for the entire batch, weirdly)
            for i in range(y_seq_length):

                write_test_logits, x, d= sess.run([model_write.logits, model_write.fc1, model_write.dec_outputs],
                    feed_dict={model_write.keep_prob:1.0, model_write.inputs:X_test[:,1:], model_write.outputs:write_dec_input})
                #print("DEC_OUT",d.shape, "HIDDEN 1", x.shape, "LOGITS", write_test_logits.shape)
                write_prediction = write_test_logits[:,-1].argmax(axis=-1)
                write_dec_input = np.hstack([write_dec_input, write_prediction[:,None]])
            write_test_new_targs, tmp = utils.lds_compare(write_dec_input[:,1:],Y_test[:,1:], Y_alt_test[:,1:], dict_num2char_y, 'test')
            lds_ratios_test[epoch] = tmp

            fullPred, fullTarg = utils.accuracy_prepare(write_dec_input[:,1:], Y_test[:,1:],dict_char2num_y, mode='test')
            print(fullTarg.shape)
            dists, write_tokenAcc = sess.run([acc_object.dists, acc_object.token_acc], 
                    feed_dict={acc_object.fullPred:fullPred, acc_object.fullTarg: fullTarg})
            write_wordAcc  = np.count_nonzero(dists==0) / len(dists) 

            print('WRITING - Accuracy on test set is for tokens{:>6.3f} and for words {:>6.3f}'.format(write_tokenAcc, write_wordAcc))

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
                    read_dec_input = np.hstack([read_dec_input, read_prediction[:,None]])



                fullPred, fullTarg = utils.accuracy_prepare(read_dec_input[:,1:], X_test[:,1:],dict_char2num_x, mode='test')
                dists, read_tokenAcc = sess.run([acc_object.dists, acc_object.token_acc], 
                        feed_dict={acc_object.fullPred:fullPred, acc_object.fullTarg: fullTarg})
                read_wordAcc  = np.count_nonzero(dists==0) / len(dists) 

                print('READING - Accuracy on test set is for tokens{:>6.3f} and for words {:>6.3f}'.format(read_tokenAcc, read_wordAcc))

                testPerf[epoch//args.print_step, 2] = read_tokenAcc
                testPerf[epoch//args.print_step, 3] = read_wordAcc


        elif regime == 'lds':

            write_dec_input = np.zeros((len(X_test), 1)) + dict_char2num_y['<GO>']
            for i in range(y_seq_length):

                write_test_logits = sess.run(model_write.logits, 
                    feed_dict={model_write.keep_prob:1.0, model_write.inputs:X_test[:,1:], model_write.outputs:write_dec_input})
                write_prediction = write_test_logits[:,-1].argmax(axis=-1)
                write_dec_input = np.hstack([write_dec_input, write_prediction[:,None]])

            # Now the generated sequence need to be lds_compareed with the alternative targets:
            write_test_new_targs, tmp = utils.lds_compare(write_dec_input[:,1:],Y_test[:,1:], Y_alt_test[:,1:], dict_num2char_y, 'test')
            lds_ratios_test[epoch] = tmp

            fullPred, fullTarg = utils.accuracy_prepare(write_dec_input[:,1:], write_test_new_targs,dict_char2num_y, mode='test')
            dists, write_tokenAcc = sess.run([acc_object.dists, acc_object.token_acc], 
                    feed_dict={acc_object.fullPred:fullPred, acc_object.fullTarg: fullTarg})

            write_wordAcc  = np.count_nonzero(dists==0) / len(dists) 


            print('WRITING - Accuracy on test set is for tokens{:>6.3f} and for words {:>6.3f}'.format(write_tokenAcc, write_wordAcc))

            testPerf[epoch//args.print_step, 0] = write_tokenAcc
            testPerf[epoch//args.print_step, 1] = write_wordAcc

            # Test READING
            if args.reading:
                read_dec_input = np.zeros((len(X_test), 1)) + dict_char2num_x['<GO>']
                read_test_new_inp = write_test_new_targs
                for i in range(x_seq_length):
                    read_test_logits = sess.run(model_read.logits, feed_dict={model_read.keep_prob:1.0, 
                            model_read.inputs:read_test_new_inp, model_read.outputs:read_dec_input})
                    read_prediction = read_test_logits[:,-1].argmax(axis=-1)
                    read_dec_input = np.hstack([read_dec_input, read_prediction[:,None]])

                fullPred, fullTarg = utils.accuracy_prepare(read_dec_input[:,1:], X_test[:,1:],dict_char2num_x, mode='test')
                dists, read_tokenAcc = sess.run([acc_object.dists, acc_object.token_acc], 
                        feed_dict={acc_object.fullPred:fullPred, acc_object.fullTarg: fullTarg})
                read_wordAcc  = np.count_nonzero(dists==0) / len(dists) 
                #read_losses[epoch//args.print_step] = read_test_loss

                print('READING - Accuracy on test set is for tokens{:>6.3f} and for words {:>6.3f}'.format(read_tokenAcc, read_wordAcc))
                testPerf[epoch//args.print_step, 2] = read_tokenAcc
                testPerf[epoch//args.print_step, 3] = read_wordAcc



        print("Time the testing took", time()-tt)

        if epoch % args.save_model == 0 and epoch > 0:
            #saver_write.save(sess, save_path + '/Model_write', global_step=epoch, write_meta_graph=True)
            #if args.reading:
            #    saver_read.save(sess, save_path + '/Model_read', global_step=epoch, write_meta_graph=True)
            np.savez(save_path + '/metrics.npz', trainPerf=trainPerf, testPerf=testPerf, lds_ratios=lds_ratios,lds_loss=lds_losses, 
                    write_loss=write_losses, read_losses=read_losses, lds_ratios_test=lds_ratios_test)
            saver.save(sess, save_path + '/my_test_model',global_step=epoch)

        elif epoch == 120:
            np.savez(save_path + '/metrics.npz', trainPerf=trainPerf, testPerf=testPerf, lds_ratios=lds_ratios,lds_loss=lds_losses, 
                    write_loss=write_losses, read_losses=read_losses, lds_ratios_test=lds_ratios_test)
            saver.save(sess, save_path + '/my_test_model',global_step=epoch)        

        # If lds learning is performed, training regime is changed to normal after half of the epochs 
        if args.learn_type == 'lds' or args.learn_type == 'intervened':

            if args.epochs // 2 == epoch and regime == 'lds':
                regime = 'normal'
                print("Training regime changed to normal\n")

        # In interleaved regime, in regular training (2nd half), every 5th epoch is again LdS epoch
        if args.learn_type == 'interleaved':

            if epoch > args.epochs // 2 and epoch % 5 == 0:
                regime = 'lds'
                print("Training regime changed to lds\n")

            elif epoch > args.epochs // 2 and regime == 'lds':
                regime = 'normal'
                print("Training regime changed back to normal\n") 

        # In intervened regime, within LdS training (1st half), every 10th epoch is a regular epoch
        elif args.learn_type == 'intervened':

            if epoch < args.epochs // 2 and epoch % 10 == 0 and epoch > 0:
                regime = 'normal'
                print("Training regime changed to normal\n")

            elif epoch < args.epochs // 2 and regime == 'normal':
                regime = 'lds'
                print("Training regime changed back to lds\n") 

        # In intervened+interleaved regime, both other regimes are combined
        elif args.learn_type == 'intervened + interleaved':

            if epoch < args.epochs // 2 and epoch % 10 == 0 and epoch > 0:
                regime = 'normal'
                print("Training regime changed to normal\n")

            elif epoch < args.epochs // 2 and regime == 'normal':
                regime = 'lds'
                print("Training regime changed back to lds\n") 

            elif epoch > args.epochs // 2 and epoch % 5 == 0:
                regime = 'lds'
                print("Training regime changed to lds\n")

            elif epoch > args.epochs // 2 and regime == 'lds':
                regime = 'normal'
                print("Training regime changed back to normal\n") 

    
    saver.save(sess, save_path + '/my_test_model',global_step=epoch)
  

    print(" Training done, model_write saved in file: %s" % save_path + ' ' + os.path.abspath(save_path))

    #np.savetxt(save_path+'/train.txt', trainPerf, delimiter=',')   
    #np.savetxt(save_path+'/test.txt', testPerf, delimiter=',')  
    np.savez(save_path + '/metrics.npz', trainPerf=trainPerf, testPerf=testPerf, lds_ratios=lds_ratios,lds_loss=lds_losses, 
        write_loss=write_losses, read_losses=read_losses, lds_ratios_test=lds_ratios_test, lt=lt)


print("Learning types were ", lt)
print("DONE!")   


