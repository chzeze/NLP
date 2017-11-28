__author__ = 'max'



import time
import sys
import argparse
from lasagne_nlp.utils import utils
import lasagne_nlp.utils.data_processor as data_processor
from lasagne_nlp.utils.objectives import crf_loss, crf_accuracy, my_crf_accuracy
import lasagne
import theano
import theano.tensor as T
from lasagne_nlp.networks.networks import build_BiLSTM_CNN_CRF

import numpy as np

def printout(words,preds,truth):
    for i in xrange(len(words)):
        print "\t".join([str(i+1),words[i],"_","_","_",preds[i],truth[i],"_","_"])


def main():
    parser = argparse.ArgumentParser(description='Tuning with bi-directional LSTM-CNN-CRF')
    parser.add_argument('--fine_tune', action='store_true', help='Fine tune the word embeddings')
    parser.add_argument('--embedding', choices=['word2vec', 'glove', 'senna', 'random'], help='Embedding for words',
                        required=True)
    parser.add_argument('--embedding_dict', default=None, help='path for embedding dict')
    parser.add_argument('--batch_size', type=int, default=10, help='Number of sentences in each batch')
    parser.add_argument('--num_units', type=int, default=100, help='Number of hidden units in LSTM')
    parser.add_argument('--num_filters', type=int, default=20, help='Number of filters in CNN')
    parser.add_argument('--learning_rate', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--decay_rate', type=float, default=0.1, help='Decay rate of learning rate')
    parser.add_argument('--grad_clipping', type=float, default=0, help='Gradient clipping')
    parser.add_argument('--gamma', type=float, default=1e-6, help='weight for regularization')
    parser.add_argument('--peepholes', action='store_true', help='Peepholes for LSTM')
    parser.add_argument('--oov', choices=['random', 'embedding'], help='Embedding for oov word', required=True)
    parser.add_argument('--update', choices=['sgd', 'momentum', 'nesterov', 'adadelta'], help='update algorithm',
                        default='sgd')
    parser.add_argument('--regular', choices=['none', 'l2'], help='regularization for training', required=True)
    parser.add_argument('--dropout', action='store_true', help='Apply dropout layers')
    parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping')
    parser.add_argument('--output_prediction', action='store_true', help='Output predictions to temp files')
    parser.add_argument('--train')  # "data/POS-penn/wsj/split1/wsj1.train.original"
    parser.add_argument('--dev')  # "data/POS-penn/wsj/split1/wsj1.dev.original"
    parser.add_argument('--test')  # "data/POS-penn/wsj/split1/wsj1.test.original"
    parser.add_argument('--realtest')
    parser.add_argument('--mymodel')

    args = parser.parse_args()

    def construct_input_layer():
        if fine_tune:
            layer_input = lasagne.layers.InputLayer(shape=(None, max_length), input_var=input_var, name='input')
            layer_embedding = lasagne.layers.EmbeddingLayer(layer_input, input_size=alphabet_size,
                                                            output_size=embedd_dim,
                                                            W=embedd_table, name='embedding')
            return layer_embedding
        else:
            layer_input = lasagne.layers.InputLayer(shape=(None, max_length, embedd_dim), input_var=input_var,
                                                    name='input')
            return layer_input

    def construct_char_input_layer():
        layer_char_input = lasagne.layers.InputLayer(shape=(None, max_sent_length, max_char_length),
                                                     input_var=char_input_var, name='char-input')
        layer_char_input = lasagne.layers.reshape(layer_char_input, (-1, [2]))
        layer_char_embedding = lasagne.layers.EmbeddingLayer(layer_char_input, input_size=char_alphabet_size,
                                                             output_size=char_embedd_dim, W=char_embedd_table,
                                                             name='char_embedding')
        layer_char_input = lasagne.layers.DimshuffleLayer(layer_char_embedding, pattern=(0, 2, 1))
        return layer_char_input

    logger = utils.get_logger("BiLSTM-CNN-CRF")
    fine_tune = args.fine_tune
    oov = args.oov
    regular = args.regular
    embedding = args.embedding
    embedding_path = args.embedding_dict
    train_path = args.train
    dev_path = args.dev
    test_path = args.test
    real_test_path = args.realtest
    update_algo = args.update
    grad_clipping = args.grad_clipping
    peepholes = args.peepholes
    num_filters = args.num_filters
    gamma = args.gamma
    output_predict = args.output_prediction
    dropout = args.dropout
    mymodel = args.mymodel
    print "Model is",mymodel,test_path,real_test_path
    
    X_train, Y_train, mask_train, X_dev, Y_dev, mask_dev, X_test, Y_test, mask_test, _X_real_test, _Y_real_test, _mask_real_test, \
    embedd_table, label_alphabet, word_alphabet, \
    C_train, C_dev, C_test, _C_real_test, char_embedd_table = data_processor.load_dataset_sequence_labeling(train_path, dev_path,
                                                                                              test_path, test_path, oov=oov,
                                                                                              fine_tune=fine_tune,
                                                                                              embedding=embedding,
                                                                                              embedding_path=embedding_path,
                                                                                                           use_character=True)

    _X_train, _Y_train, _mask_train, _X_dev, _Y_dev, _mask_dev, _X_test, _Y_test, _mask_test, X_real_test, Y_real_test, mask_real_test, \
    _embedd_table, _label_alphabet, _word_alphabet, \
    _C_train, _C_dev, _C_test, C_real_test, _char_embedd_table = data_processor.load_dataset_sequence_labeling(train_path, dev_path,
                                                            test_path, real_test_path, oov=oov,fine_tune=fine_tune,
                                                            embedding=embedding,
                                                            embedding_path=embedding_path,use_character=True)

    #print _C_train.shape,_C_dev.shape,_C_test.shape,C_real_test.shape; #sys.exit(1)
    my_size = data_processor.MAX_LENGTH_TRAIN
    my_size = data_processor.MY_MAX_LENGTH
    #my_size = data_processor.MAX_LENGTH_DEV
    print "\tMYSIZE",my_size,C_real_test.shape,C_test.shape,C_train.shape
    num_labels = label_alphabet.size() - 1

    logger.info("constructing network...")
    # create variables
    target_var = T.imatrix(name='targets')
    mask_var = T.matrix(name='masks', dtype=theano.config.floatX)
    if fine_tune:
        input_var = T.imatrix(name='inputs')
        num_data, max_length = X_train.shape
        alphabet_size, embedd_dim = embedd_table.shape
    else:
        input_var = T.tensor3(name='inputs', dtype=theano.config.floatX)
        num_data, max_length, embedd_dim = X_train.shape
    char_input_var = T.itensor3(name='char-inputs')
    num_data_char, max_sent_length, max_char_length = C_train.shape
    char_alphabet_size, char_embedd_dim = char_embedd_table.shape
    assert (max_length == max_sent_length)
    assert (num_data == num_data_char)

    # construct input and mask layers
    layer_incoming1 = construct_char_input_layer()
    layer_incoming2 = construct_input_layer()

    layer_mask = lasagne.layers.InputLayer(shape=(None, max_length), input_var=mask_var, name='mask')

    # construct bi-rnn-cnn
    num_units = args.num_units

    bi_lstm_cnn_crf = build_BiLSTM_CNN_CRF(layer_incoming1, layer_incoming2, num_units, num_labels, mask=layer_mask,
                                           grad_clipping=grad_clipping, peepholes=peepholes, num_filters=num_filters,
                                           dropout=dropout)
#    bi_lstm_cnn_crf = None

    logger.info("Network structure: hidden=%d, filter=%d" % (num_units, num_filters))

    # compute loss
    num_tokens = mask_var.sum(dtype=theano.config.floatX)

    # get outpout of bi-lstm-cnn-crf shape [batch, length, num_labels, num_labels]
    energies_train = lasagne.layers.get_output(bi_lstm_cnn_crf)
    energies_eval = lasagne.layers.get_output(bi_lstm_cnn_crf, deterministic=True)

    loss_train = crf_loss(energies_train, target_var, mask_var).mean()
    loss_eval = crf_loss(energies_eval, target_var, mask_var).mean()
    # l2 regularization?
    if regular == 'l2':
        l2_penalty = lasagne.regularization.regularize_network_params(bi_lstm_cnn_crf, lasagne.regularization.l2)
        loss_train = loss_train + gamma * l2_penalty

    _, corr_train = crf_accuracy(energies_train, target_var)
    corr_train = (corr_train * mask_var).sum(dtype=theano.config.floatX)
    prediction_eval, corr_eval = crf_accuracy(energies_eval, target_var)
    corr_eval = (corr_eval * mask_var).sum(dtype=theano.config.floatX)

    # Create update expressions for training.
    # hyper parameters to tune: learning rate, momentum, regularization.
    batch_size = args.batch_size
    learning_rate = 1.0 if update_algo == 'adadelta' else args.learning_rate
    decay_rate = args.decay_rate
    momentum = 0.9
    params = lasagne.layers.get_all_params(bi_lstm_cnn_crf, trainable=True)
    updates = utils.create_updates(loss_train, params, update_algo, learning_rate, momentum=momentum)

    # Compile a function performing a training step on a mini-batch
    train_fn = theano.function([input_var, target_var, mask_var, char_input_var], [loss_train, corr_train, num_tokens],
                               updates=updates)
    # Compile a second function evaluating the loss and accuracy of network
    eval_fn = theano.function([input_var, target_var, mask_var, char_input_var],
                              [loss_eval, corr_eval, num_tokens, prediction_eval])
    my_prediction_eval = my_crf_accuracy(energies_eval)
    my_eval_fn = theano.function([input_var,mask_var,char_input_var],[my_prediction_eval])

    # Finally, launch the training loop.
    logger.info(
        "Start training: %s with regularization: %s(%f), dropout: %s, fine tune: %s (#training data: %d, batch size: %d, clip: %.1f, peepholes: %s)..." \
        % (
            update_algo, regular, (0.0 if regular == 'none' else gamma), dropout, fine_tune, num_data, batch_size,
            grad_clipping,
            peepholes))
    num_batches = num_data / batch_size
    num_epochs = 1000
    best_loss = 1e+12
    best_acc = 0.0
    best_epoch_loss = 0
    best_epoch_acc = 0
    best_loss_test_err = 0.
    best_loss_test_corr = 0.
    best_acc_test_err = 0.
    best_acc_test_corr = 0.
    stop_count = 0
    lr = learning_rate
    patience = args.patience
    print "#LOADING MODEL"
    #np.savez("model.npz",*lasagne.layers.get_all_param_values(bi_lstm_cnn_crf))
    # just load the data, see here:
    # https://github.com/Lasagne/Lasagne/blob/master/examples/mnist.py
    #try: mymodel = sys.argv[1]
    #except IndexError:
    #    mymodel = "models.npz"
    with np.load(mymodel) as f:
	param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    lasagne.layers.set_all_param_values(bi_lstm_cnn_crf,param_values)
    correct = 0
    total = 0
    print dir(bi_lstm_cnn_crf)
    #print bi_lstm_cnn_crf.predict([1,2,3,4])
    #sys.exit(1)
    print X_real_test.shape,Y_real_test.shape,C_real_test.shape,mask_real_test.shape
    # that's a stupid hack
    #C_real_test = C_real_test[:len(X_real_test)]

    #print X_real_test[0:1]
    #print my_eval_fn(X_real_test[0:1],mask_real_test[0:1],C_real_test[0:1])
    #sys.exit(1)
    for batch in utils.iterate_minibatches(X_real_test, Y_real_test, masks=mask_real_test, char_inputs=C_real_test,
                                                   batch_size=batch_size):
                inputs, targets, masks, char_inputs = batch
                #print inputs,targets,masks,char_inputs; sys.exit(1)
                err, corr, num, predictions = eval_fn(inputs, targets, masks, char_inputs)
                #print predictions # SE                                                                                                  
                input_clear = [word_alphabet.get_instance(y) for x in inputs for y in list(x)] # predictions,dir(label_alphabet),label_alphabet.get_instance(4) # SE                                                                                                             
                target_clear = [label_alphabet.get_instance(y+1) for x in targets for y in list(x)]
                target_clear_pred = [label_alphabet.get_instance(y+1) for x in predictions for y in list(x)] # SE                        
                #print my_size                                                                                                           
                # comment this out                                                                                                       
                #my_size = 557                                                                                                           
                #print input_clear; sys.exit(1)                                                                                          
                for ii in range(batch_size):
                  Z = input_clear[ii*my_size:(ii+1)*my_size]
                  if len(Z)==0: continue
                  try:
                    size = Z.index(None)
                  except ValueError:
                    size = my_size
                  #print size                                                                                                            
                  itruth = input_clear[ii*my_size:(ii+1)*my_size][:size]
                  otruth = filter(lambda z: z!="EMPTY",target_clear[ii*my_size:(ii+1)*my_size][:size])
                  opred = filter(lambda z: z!="EMPTY",target_clear_pred[ii*my_size:(ii+1)*my_size][:size])
		  total += len(opred)
		  correct += len(filter(lambda x: x==True,[otruth[jj]==opred[jj] for jj in xrange(len(opred))]))
                  if otruth==opred:
                        #test_err_sentence += 1
                        #print "CORRECT",itruth,otruth,opred                                                                             
                        #print "#CORRECT %%%"
                        printout(itruth,opred,otruth)
                        print
                  else:
                        #print "#WRONG %%%" #itruth,otruth,opred                                                                          
                        printout(itruth,opred,otruth)
                        print

    print correct,total,correct*1.0/total
    #sys.exit(1)


if __name__ == '__main__':
    main()
