import numpy as np
import argparse
import importlib
import random
import os
import tensorflow as tf
from flearn.utils.model_utils import read_data

# GLOBAL PARAMETERS
OPTIMIZERS = ['qffedsgd', 'qffedavg', 'afl', 'maml']
DATASETS = [ 'synthetic', 'vehicle', 'sent140', 'shakespeare',
'synthetic_iid', 'synthetic_hybrid', 
'fmnist', 'adult', 'omniglot']   # fmnist: fashion mnist used in the AFL paper


MODEL_PARAMS = {
    'adult.lr': (2, ), # num_classes,
    'adult.lr_afl': (2, ), # num_classes,
    'sent140.stacked_lstm': (25, 2, 100), # seq_len, num_classes, num_hidden 
    'fmnist.lr': (3,), # num_classes
    'mnist.cnn': (10,),  # num_classes
    'shakespeare.stacked_lstm': (80, 80, 256), # seq_len, num_class num_hidden
    'synthetic.mclr': (10, ), # num_classes
    'vehicle.svm':(2, ), # num_classes
}


def read_options():
    ''' Parse command line arguments or load defaults '''
    parser = argparse.ArgumentParser()

    parser.add_argument('--optimizer',
                    help='name of optimizer;',
                    type=str,
                    choices=OPTIMIZERS,
                    default='qffedavg')
    parser.add_argument('--dataset',
                    help='name of dataset;',
                    type=str,
                    choices=DATASETS,
                    default='nist')
    parser.add_argument('--model',
                    help='name of model;',
                    type=str,
                    default='stacked_lstm.py')
    parser.add_argument('--num_rounds',
                    help='number of rounds to simulate;',
                    type=int,
                    default=-1)
    parser.add_argument('--eval_every',
                    help='evaluate every ____ rounds;',
                    type=int,
                    default=-1)
    parser.add_argument('--clients_per_round',
                    help='number of clients trained per round;',
                    type=int,
                    default=-1)
    parser.add_argument('--batch_size',
                    help='batch size when clients train on data;',
                    type=int,
                    default=10)
    parser.add_argument('--num_epochs', 
                    help='number of epochs when clients train on data;',
                    type=int,
                    default=1) 
    parser.add_argument('--learning_rate',
                    help='learning rate for inner solver;',
                    type=float,
                    default=0.003)
    parser.add_argument('--seed',
                    help='seed for random initialization;',
                    type=int,
                    default=0)
    parser.add_argument('--sampling',
                    help='client sampling methods',
                    type=int,
                    default='5') # uniform sampling + weighted average
    parser.add_argument('--q',
                    help='reweighting factor',
                    type=float,
                    default='0.0') # no weighting, the same as fedavg
    parser.add_argument('--output',
                    help='file to save the final accuracy across all devices',
                    type=str,
                    default='output_accu') 
    parser.add_argument('--learning_rate_lambda',
                    help='learning rate for lambda in agnostic flearn',
                    type=float,
                    default=0)
    parser.add_argument('--log_interval',
                    help='intervals (how many rounds) to output accuracy distribution (data dependent',
                    type=int,
                    default=10)
    parser.add_argument('--data_partition_seed',
                    help='seed for splitting data into train/test/validation',
                    type=int,
                    default=1)
    parser.add_argument('--static_step_size',
                    help='whether to use our method or use a best tuned step size FedSGD to solve q-FFL',
                    type=int,
                    default=0)  # default is using our method
    parser.add_argument('--track_individual_accuracy',
                    help='whether to track each device\'s accuracy, only true when comparing with AFL',
                    type=int,
                    default=0)  
    parser.add_argument('--held_out',
                    help="number of held out devices/tasks",
                    type=int,
                    default=0)
    parser.add_argument('--num_fine_tune',
                    help="number of fine-tuning iterations",
                    type=int,
                    default=0)
    parser.add_argument('--with_maml',
                    help="whether to learn better intializations or use finetuning baseline",
                    type=int,
                    default=0)

    try: parsed = vars(parser.parse_args())
    except IOError as msg: parser.error(str(msg))

    # load selected model
    if parsed['dataset'].startswith("synthetic"):  # all synthetic datasets use the same model
        model_path = '%s.%s.%s.%s' % ('flearn', 'models', 'synthetic', parsed['model'])
    else:
        model_path = '%s.%s.%s.%s' % ('flearn', 'models', parsed['dataset'], parsed['model'])

    mod = importlib.import_module(model_path)
    learner = getattr(mod, 'Model')

    # load selected trainer
    opt_path = 'flearn.trainers.%s' % parsed['optimizer']
    mod = importlib.import_module(opt_path)
    optimizer = getattr(mod, 'Server')

    # add selected model parameter
    parsed['model_params'] = MODEL_PARAMS['.'.join(model_path.split('.')[2:])]


    # print and return
    maxLen = max([len(ii) for ii in parsed.keys()]);
    fmtString = '\t%' + str(maxLen) + 's : %s';
    print('Arguments:')
    for keyPair in sorted(parsed.items()): print(fmtString % keyPair)

    return parsed, learner, optimizer

def main():
    """
    Main function.

    Args:
    """
    # suppress tf warnings
    tf.logging.set_verbosity(tf.logging.WARN)
    
    # parse command line arguments
    options, learner, optimizer = read_options()

    # read data
    train_path = os.path.join('data', options['dataset'], 'data', 'train')
    test_path = os.path.join('data', options['dataset'], 'data', 'test')
    dataset = read_data(train_path, test_path)

    # call appropriate trainer
    t = optimizer(options, learner, dataset)
    t.train()
    
if __name__ == '__main__':
    main()





