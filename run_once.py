from Faraone_TF import run_MLP
import hyperopt
import tensorflow as tf
import re
import pickle
import csv_reader
import config_reader
import utils


def objective(args):
    config = config_reader.read_config(utils.abs_path_of("config/default.ini"))
    params = {}

    params['l1_reg'] = args['l1_reg']
    params['l2_reg'] = args['l2_reg']
    params['num_layers'] = int(args['num_layers'])
    params['layer_size'] = int(args['layer_size'])
    params['learning_rate'] = args['learning_rate']
    params['batch_size'] = args['batch_size']
    params['dropout_keep_probability'] = args['dropout_keep_probability']
    params['validation_window'] = args['validation_window']

    trows = csv_reader.read_csv_dataframe(config.get_rel_path("PATHS", "training_file"))
    vrows = csv_reader.read_csv_dataframe(config.get_rel_path("PATHS", "validation_file"))

    with tf.Graph().as_default():
        loss = run_MLP(params, trows, vrows)

    return loss


def optimize():
    save_trial = 1
    max_trials = 1

    space = {
        'l1_reg': 0.1335267529016616,
        'l2_reg': 0.11497692967392144,
        'learning_rate': 0.00005280539672336577,
        'num_layers': 1,
        'layer_size': 10,
        'batch_size': 512,
        'dropout_keep_probability': 0.8846418566190806,
        'validation_window': 10
    }
    import time
    tick = time.time()
    objective(space)
    print(time.time() - tick)


if __name__ == '__main__':
    optimize()

