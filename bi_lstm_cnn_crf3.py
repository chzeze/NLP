# -*- coding: UTF-8 -*-
__author__ = 'max'

import time
import sys
import argparse
from lasagne_nlp.utils import utils
import lasagne_nlp.utils.data_processor_orig as data_processor
from lasagne_nlp.utils.objectives import crf_loss, crf_accuracy
import lasagne
import theano
import theano.tensor as T
from lasagne_nlp.networks.networks import build_BiLSTM_CNN_CRF

import numpy as np
from sklearn.metrics import f1_score
from sklearn import metrics


def printout(words, preds, truth):
    for i in xrange(len(words)):
        print "\t".join([str(i + 1), words[i], "_", "_", "_", preds[i], truth[i], "_", "_"])


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
    parser.add_argument('--output_prediction', default='true', action='store_true',
                        help='Output predictions to temp files')
    parser.add_argument('--train')  # "data/POS-penn/wsj/split1/wsj1.train.original"
    parser.add_argument('--dev')  # "data/POS-penn/wsj/split1/wsj1.dev.original"
    parser.add_argument('--test')  # "data/POS-penn/wsj/split1/wsj1.test.original"
    parser.add_argument("--model")  # model name

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
    update_algo = args.update
    grad_clipping = args.grad_clipping
    peepholes = args.peepholes
    num_filters = args.num_filters
    gamma = args.gamma
    output_predict = args.output_prediction
    dropout = args.dropout
    modelname = args.model
    # 读取数据训练集，dev集，测试集
    X_train, Y_train, mask_train, X_dev, Y_dev, mask_dev, X_test, Y_test, mask_test, \
    embedd_table, label_alphabet, word_alphabet, \
    C_train, C_dev, C_test, char_embedd_table = data_processor.load_dataset_sequence_labeling(train_path, dev_path,
                                                                                              test_path, oov=oov,
                                                                                              fine_tune=fine_tune,
                                                                                              embedding=embedding,
                                                                                              embedding_path=embedding_path,
                                                                                              use_character=True)
    print 'label_alphabet'
    for i in range(label_alphabet.size()):
        print i
        print label_alphabet.get_instance(i)

    # print Y_test, Y_test.shape; sys.exit(1)
    my_size = data_processor.MAX_LENGTH_TRAIN
    my_size = data_processor.MY_MAX_LENGTH
    print "\tMY_SIZE", my_size, data_processor.MAX_LENGTH_TRAIN
    # my_size = data_processor.MAX_LENGTH_DEV
    print "\tMYSIZE", my_size
    num_labels = label_alphabet.size() - 1
    # 构建网络
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
    # 构建输入层
    # construct input and mask layers
    logger.info("construct input and mask layers...")
    layer_incoming1 = construct_char_input_layer()
    layer_incoming2 = construct_input_layer()

    layer_mask = lasagne.layers.InputLayer(shape=(None, max_length), input_var=mask_var, name='mask')

    # construct bi-rnn-cnn
    logger.info("construct bi-rnn-cnn...")
    num_units = args.num_units

    bi_lstm_cnn_crf = build_BiLSTM_CNN_CRF(layer_incoming1, layer_incoming2, num_units, num_labels, mask=layer_mask,
                                           grad_clipping=grad_clipping, peepholes=peepholes, num_filters=num_filters,
                                           dropout=dropout)

    logger.info("Network structure: hidden=%d, filter=%d" % (num_units, num_filters))

    # compute loss
    num_tokens = mask_var.sum(dtype=theano.config.floatX)

    # get outpout of bi-lstm-cnn-crf shape [batch, length, num_labels, num_labels]
    energies_train = lasagne.layers.get_output(bi_lstm_cnn_crf)
    energies_eval = lasagne.layers.get_output(bi_lstm_cnn_crf, deterministic=True)

    loss_train = crf_loss(energies_train, target_var, mask_var).mean()
    # print loss_train; sys.exit(1)
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

    # Finally, launch the training loop.
    logger.info(
        "Start training: %s with regularization: %s(%f), dropout: %s, fine tune: %s (#training data: %d, batch size: %d, clip: %.1f, peepholes: %s)..." \
        % (
            update_algo, regular, (0.0 if regular == 'none' else gamma), dropout, fine_tune, num_data, batch_size,
            grad_clipping,
            peepholes))
    num_batches = num_data / batch_size
    num_epochs = 1000
    # num_epochs = 1
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
    for epoch in range(1, num_epochs + 1):

        print 'Epoch %d (learning rate=%.4f, decay rate=%.4f): ' % (epoch, lr, decay_rate)
        train_err = 0.0
        train_corr = 0.0
        train_total = 0
        train_inst = 0
        start_time = time.time()
        num_back = 0
        train_batches = 0
        for batch in utils.iterate_minibatches(X_train, Y_train, masks=mask_train, char_inputs=C_train,
                                               batch_size=batch_size, shuffle=True):
            inputs, targets, masks, char_inputs = batch
            err, corr, num = train_fn(inputs, targets, masks, char_inputs)
            train_err += err * inputs.shape[0]
            train_corr += corr
            train_total += num
            train_inst += inputs.shape[0]
            train_batches += 1
            time_ave = (time.time() - start_time) / train_batches
            time_left = (num_batches - train_batches) * time_ave

            # update log
            sys.stdout.write("\b" * num_back)
            log_info = 'train: %d/%d loss: %.4f, acc: %.2f%%, time left (estimated): %.2fs' % (
                min(train_batches * batch_size, num_data), num_data,
                train_err / train_inst, train_corr * 100 / train_total, time_left)
            sys.stdout.write(log_info)
            num_back = len(log_info)
        # update training log after each epoch
        assert train_inst == num_data
        sys.stdout.write("\b" * num_back)
        print 'train: %d/%d loss: %.4f, acc: %.2f%%, time: %.2fs' % (
            min(train_batches * batch_size, num_data), num_data,
            train_err / num_data, train_corr * 100 / train_total, time.time() - start_time)

        # evaluate performance on dev data
        dev_err = 0.0
        dev_corr = 0.0
        dev_total = 0
        dev_inst = 0
        test_err_sentence = 0
        my_f1 = {}
        my_prs = []
        my_trs = []
        for batch in utils.iterate_minibatches(X_dev, Y_dev, masks=mask_dev, char_inputs=C_dev, batch_size=batch_size):
            inputs, targets, masks, char_inputs = batch
            err, corr, num, predictions = eval_fn(inputs, targets, masks, char_inputs)

            # print "-->HERE COMES THE PREDS",predictions
            # print predictions.shape,type(predictions);
            for i in xrange(batch_size):
                try:
                    input_clear = [word_alphabet.get_instance(y) for y in list(inputs[i, :])]
                except IndexError:
                    continue
                if len(input_clear) == 0: continue
                try:
                    dev_size = input_clear.index(None)
                except ValueError:
                    dev_size = my_size
                    # print dev_size
                my_trs += list(targets[i, :dev_size])
                my_prs += list(predictions[i, :dev_size])
                # for j in xrange(len(targets[i,:dev_size])):
            #  pr = predictions[i,j]
            #  tr = targets[i,j]
            #  my_f1[(pr,tr)] = my_f1.get((pr,tr),0)+1
            # input_clear = [word_alphabet.get_instance(y) for y in list(inputs[0,:])]
            # print input_clear
            # my_f1 = f1_score(my_trs,my_prs,average="macro")
            # print [label_alphabet.get_instance(y+1) for y in list(targets[0,:])]
            # print targets[0,:],predictions[0,:],inputs.shape[0],my_f1; sys.exit(1)
            # print err,inputs.shape[0]

            dev_err += err * inputs.shape[0]
            dev_corr += corr
            dev_total += num
            dev_inst += inputs.shape[0]
            if output_predict:
                utils.output_predictions(predictions, targets, masks, 'tmp/dev%d' % epoch, label_alphabet,
                                         is_flattened=False)

        dev_f1 = f1_score(my_trs, my_prs, average="macro")
        classify_report = metrics.classification_report(my_trs, my_prs)
        print 'dev classify_report'
        print classify_report
        print 'dev loss: %.4f, corr: %d, total: %d, acc: %.2f%%, f1: %.4f' % (
            dev_err / dev_inst, dev_corr, dev_total, dev_corr * 100 / dev_total, dev_f1)

        # CHANGE THIS IF NECESSARY
        # MODEL SELECTION ON DEV CRITERION, SE
        useF1 = True
        useLoss = False
        criterion = dev_f1 if useF1 else dev_corr / dev_total

        if best_loss < dev_err and best_acc > criterion:
            stop_count += 1
        else:
            update_loss = False
            update_acc = False
            stop_count = 0
            if best_loss > dev_err:
                update_loss = True
                best_loss = dev_err
                best_epoch_loss = epoch
            if best_acc < criterion:
                update_acc = True
                best_acc = criterion
                best_epoch_acc = epoch
            else:
                if useLoss == False:
                    continue

            # evaluate on test data when better performance detected
            test_err = 0.0
            test_corr = 0.0
            test_total = 0
            test_inst = 0
            test_err_sentence = 0
            test_sentences = 0
            print "#SAVING MODEL"
            np.savez(modelname, *lasagne.layers.get_all_param_values(bi_lstm_cnn_crf))
            test_prs = []
            test_trs = []
            for batch in utils.iterate_minibatches(X_test, Y_test, masks=mask_test, char_inputs=C_test,
                                                   batch_size=batch_size):
                inputs, targets, masks, char_inputs = batch
                err, corr, num, predictions = eval_fn(inputs, targets, masks, char_inputs)

                # print "-->HERE COMES THE PREDS",predictions
                # print predictions.shape,type(predictions);
                for i in xrange(batch_size):
                    try:
                        input_clear = [word_alphabet.get_instance(y) for y in list(inputs[i, :])]
                    except IndexError:
                        continue
                    if len(input_clear) == 0: continue
                    try:
                        test_size = input_clear.index(None)
                    except ValueError:
                        test_size = my_size
                        # print dev_size
                    test_trs += list(targets[i, :test_size])
                    test_prs += list(predictions[i, :test_size])

                # print predictions # SE
                # print "AAA",inputs[0],len(inputs[0]),inputs[0][0]
                input_clear = [word_alphabet.get_instance(y) for x in inputs for y in
                               list(x)]  # predictions,dir(label_alphabet),label_alphabet.get_instance(4) # SE
                target_clear = [label_alphabet.get_instance(y + 1) for x in targets for y in list(x)]
                target_clear_pred = [label_alphabet.get_instance(y + 1) for x in predictions for y in list(x)]  # SE

                # print my_size
                # comment this out
                # my_size = 652
                # print input_clear; sys.exit(1)
                # print input_clear
                for ii in range(batch_size):
                    Z = input_clear[ii * my_size:(ii + 1) * my_size]
                    if len(Z) == 0: continue
                    try:
                        size = Z.index(None)
                    except ValueError:
                        size = my_size
                        # print size
                    itruth = input_clear[ii * my_size:(ii + 1) * my_size][:size]
                    EMPTY = "EMPTY"
                    EMPTY = "EMPTY_EMPTY"
                    otruth = filter(lambda z: z != EMPTY, target_clear[ii * my_size:(ii + 1) * my_size][:size])
                    opred = filter(lambda z: z != EMPTY, target_clear_pred[ii * my_size:(ii + 1) * my_size][:size])
                    if otruth == opred:
                        test_err_sentence += 1
                        # print "CORRECT",itruth,otruth,opred
                        print "#CORRECT %%%", len(itruth), len(opred), len(otruth)
                        printout(itruth, opred, otruth)
                        print
                    else:
                        print "#WRONG %%%"  # ,itruth,otruth,opred
                        printout(itruth, opred, otruth)
                        print
                    test_sentences += 1

                test_err += err * inputs.shape[0]
                test_corr += corr
                test_total += num
                test_inst += inputs.shape[0]
                if output_predict:
                    utils.output_predictions(predictions, targets, masks, 'tmp/test%d' % epoch, label_alphabet,
                                             is_flattened=False)

            test_f1 = f1_score(test_trs, test_prs, average="macro")
            test_classify_report = metrics.classification_report(test_trs, test_prs)
            print 'label_alphabet'
            for i in range(label_alphabet.size()):
                print i
                print label_alphabet.get_instance(i)
            print 'Epoch %d test classify_report' % epoch
            print test_classify_report
            print 'test loss: %.4f, corr: %d, total: %d, acc: %.2f%% f1: %.4f' % (
                test_err / test_inst, test_corr, test_total,
                test_corr * 100 / test_total,
                test_f1), test_err_sentence * 1.0 / test_sentences, test_err_sentence, test_sentences

            if update_loss:
                best_loss_test_err = test_err
                best_loss_test_corr = test_corr
            if update_acc:
                best_acc_test_err = test_err
                best_acc_test_corr = test_corr

        # stop if dev acc decrease 3 time straightly.
        if stop_count == patience:
            break

        # re-compile a function with new learning rate for training
        if update_algo != 'adadelta':
            lr = learning_rate / (1.0 + epoch * decay_rate)
            updates = utils.create_updates(loss_train, params, update_algo, lr, momentum=momentum)
            train_fn = theano.function([input_var, target_var, mask_var, char_input_var],
                                       [loss_train, corr_train, num_tokens],
                                       updates=updates)

    # print best performance on test data.
    logger.info("final best loss test performance (at epoch %d)" % best_epoch_loss)
    print 'test loss: %.4f, corr: %d, total: %d, acc: %.2f%%' % (
        best_loss_test_err / test_inst, best_loss_test_corr, test_total, best_loss_test_corr * 100 / test_total)
    logger.info("final best acc test performance (at epoch %d)" % best_epoch_acc)
    print 'test loss: %.4f, corr: %d, total: %d, acc: %.2f%%' % (
        best_acc_test_err / test_inst, best_acc_test_corr, test_total, best_acc_test_corr * 100 / test_total)


def test():
    energies_var = T.tensor4('energies', dtype=theano.config.floatX)
    targets_var = T.imatrix('targets')
    masks_var = T.matrix('masks', dtype=theano.config.floatX)
    layer_input = lasagne.layers.InputLayer([2, 2, 3, 3], input_var=energies_var)
    out = lasagne.layers.get_output(layer_input)
    loss = crf_loss(out, targets_var, masks_var)
    prediction, acc = crf_accuracy(energies_var, targets_var)

    fn = theano.function([energies_var, targets_var, masks_var], [loss, prediction, acc])

    energies = np.array([[[[10, 15, 20], [5, 10, 15], [3, 2, 0]], [[5, 10, 1], [5, 10, 1], [5, 10, 1]]],
                         [[[5, 6, 7], [2, 3, 4], [2, 1, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]]], dtype=np.float32)

    targets = np.array([[0, 1], [0, 2]], dtype=np.int32)

    masks = np.array([[1, 1], [1, 0]], dtype=np.float32)

    l, p, a = fn(energies, targets, masks)
    print l
    print p
    print a


if __name__ == '__main__':
    main()
