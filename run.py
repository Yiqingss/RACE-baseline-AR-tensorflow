# -*- coding:utf8 -*-
# ==============================================================================
# RACE baseline AR & GA
# tensorflow version
# ==============================================================================
"""
This module prepares and runs the whole system.
"""

import sys
#reload(sys)
#sys.setdefaultencoding('utf8')
#sys.path.append('..')
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import pickle
import argparse
import logging
from dataset import Dataset
from vocab import Vocab
from model import RCModel


def parse_args():
    """
    Parses command line arguments.
    """
    parser = argparse.ArgumentParser('Reading Comprehension on RACE dataset')
    parser.add_argument('--prepare', action='store_true',
                        help='create the directories, prepare the vocabulary and embeddings')
    parser.add_argument('--train', action='store_true',
                        help='train the model')
    parser.add_argument('--evaluate_test', action='store_true',
                        help='evaluate the model on test set')
    parser.add_argument('--debug', action='store_true',
                        help='debug')
    parser.add_argument('--score', action='store_true',
                        help='find scores')
    parser.add_argument('--gpu', type=str, default='0',
                        help='specify gpu device')
    parser.add_argument('--restore_epoch', type=str, default='1',
                        help='restore model')

    train_settings = parser.add_argument_group('train settings')
    train_settings.add_argument('--optim', default='adam',
                                help='optimizer type')
    train_settings.add_argument('--learning_rate', type=float, default=0.001,
                                help='learning rate')
    train_settings.add_argument('--weight_decay', type=float, default=0,
                                help='weight decay')
    train_settings.add_argument('--dropout_keep_prob', type=float, default=0.3,
                                help='dropout keep rate')
    train_settings.add_argument('--batch_size', type=int, default=32,
                                help='train batch size')
    train_settings.add_argument('--epochs', type=int, default=30,
                                help='train epochs')

    model_settings = parser.add_argument_group('model settings')
    model_settings.add_argument('--embed_size', type=int, default=100,
                                help='size of the embeddings')
    model_settings.add_argument('--hidden_size', type=int, default=256,
                                help='size of LSTM hidden units')

    path_settings = parser.add_argument_group('path settings')
    path_settings.add_argument('--train_files', nargs='+',
                               default=['data/train_middle.json'] + ['data/train_high.json'],
                               help='list of files that contain the preprocessed train data')
    path_settings.add_argument('--dev_files', nargs='+',
                               default=['data/dev_middle.json'] + ['data/dev_high.json'],
                               help='list of files that contain the preprocessed dev data')
    path_settings.add_argument('--test_files', nargs='+',
                               default=['data/test_middle.json'] + ['data/test_high.json'],
                               help='list of files that contain the preprocessed test data')
    path_settings.add_argument('--vocab_dir', default='vocab/',
                               help='the dir to save vocabulary')
    path_settings.add_argument('--model_dir', default='models/',
                               help='the dir to store models')
    path_settings.add_argument('--result_dir', default='results/',
                               help='the dir to output the results')
    path_settings.add_argument('--summary_dir', default='summary/',
                               help='the dir to write tensorboard summary')
    path_settings.add_argument('--log_path',
                               help='path of the log file. If not set, logs are printed to console')
    return parser.parse_args()


def prepare(args):
    """
    checks data, creates the directories, prepare the vocabulary and embeddings
    """
    logger = logging.getLogger("brc")
    logger.info('Checking the data files...')
    for data_path in args.train_files + args.dev_files + args.test_files:
        assert os.path.exists(data_path), '{} file does not exist.'.format(data_path)
    logger.info('Preparing the directories...')
    for dir_path in [args.vocab_dir, args.model_dir, args.result_dir, args.summary_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    logger.info('Building vocabulary...')
    pro_data = Dataset(args.train_files, args.dev_files, args.test_files)
    vocab = Vocab(lower=True)
    for word in pro_data.word_iter('train'):
        vocab.add(word)

    unfiltered_vocab_size = vocab.size()
    vocab.filter_tokens_by_cnt(min_cnt=4)
    filtered_num = unfiltered_vocab_size - vocab.size()
    logger.info('After filter {} tokens, the final vocab size is {}'.format(filtered_num,
                                                                            vocab.size()))

    logger.info('Assigning embeddings...')
    #vocab.randomly_init_embeddings(args.embed_size)
    vocab.load_pretrained_embeddings('vocab/glove.6B.100d.txt')

    logger.info('Saving vocab...')
    with open(os.path.join(args.vocab_dir, 'vocab.data'), 'wb') as fout:
        pickle.dump(vocab, fout)

    logger.info('Done with preparing!')


def train(args):
    """
    trains the reading comprehension model
    """
    logger = logging.getLogger("brc")
    logger.info('Load data_set and vocab...')
    with open(os.path.join(args.vocab_dir, 'vocab.data'), 'rb') as fin:
        vocab = pickle.load(fin)
    rc_data = Dataset(train_files=args.train_files, dev_files=args.dev_files)
    logger.info('Converting text into ids...')
    rc_data.convert_to_ids(vocab)
    logger.info('Initialize the model...')
    rc_model = RCModel(vocab, args)
    logger.info('Training the model...')
    rc_model.train(rc_data, args.epochs, args.batch_size, save_dir=args.model_dir,
                   dropout_keep_prob=args.dropout_keep_prob)
    logger.info('Done with model training!')


def evaluate_test(args):
    """
    evaluate the trained model on test files
    """
    logger = logging.getLogger("brc")
    logger.info('Load data_set and vocab...')
    with open(os.path.join(args.vocab_dir, 'vocab.data'), 'rb') as fin:
        vocab = pickle.load(fin)
    assert len(args.test_files) > 0, 'No dev files are provided.'
    pro_data = Dataset(test_files=args.test_files)
    logger.info('Converting text into ids...')
    pro_data.convert_to_ids(vocab)
    logger.info('Restoring the model...')
    rc_model = RCModel(vocab, args)
    rc_model.restore(model_dir=args.model_dir, model_prefix=args.restore_epoch)
    logger.info('Evaluating the model on test set...')
    test_batches = pro_data.gen_mini_batches('test', args.batch_size,
                                            pad_id=vocab.get_id(vocab.pad_token), shuffle=False)
    test_loss, test_acc = rc_model.evaluate(
        test_batches, result_dir=args.result_dir, result_prefix='test.predicted')
    logger.info('Accuracy on test set: {}'.format(test_acc))



def debug(args):
    """
    small batch used to debug
    """
    logger = logging.getLogger("brc")
    logger.info('Load data_set and vocab...')
    with open(os.path.join(args.vocab_dir, 'vocab.data'), 'rb') as fin:
        vocab = pickle.load(fin)
    rc_data = Dataset(train_files=args.test_files, dev_files=args.dev_files)
    logger.info('Converting text into ids...')
    rc_data.convert_to_ids(vocab)
    logger.info('Initialize the model...')
    rc_model = RCModel(vocab, args)
    logger.info('Training the model...')
    rc_model.train(rc_data, args.epochs, args.batch_size, save_dir=args.model_dir,
                   dropout_keep_prob=args.dropout_keep_prob)
    logger.info('Done with model training!')



def run():
    """
    Prepares and runs the whole system.
    """
    args = parse_args()

    logger = logging.getLogger("brc")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    if args.log_path:
        file_handler = logging.FileHandler(args.log_path)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    else:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    logger.info('Running with args : {}'.format(args))

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if args.prepare:
        prepare(args)
    if args.train:
        train(args)
    if args.evaluate_test:
        evaluate_test(args)
    if args.evaluate_dev:
        evaluate_dev(args)
    if args.debug:
        debug(args)
    if args.score:
        predict_score(args)

if __name__ == '__main__':
    run()
