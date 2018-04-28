import config_reader
import csv_reader
import utils
from Faraone_TF import run_MLP
import hyperopt
import tensorflow as tf
import re


def objective(args):
    config = config_reader.read_config(utils.abs_path_of("config/default.ini"))
    params = {}

    params['l1_reg'] = args['l1_reg']
    params['l2_reg'] = args['l2_reg']
    params['num_layers'] = args['num_layers']
    params['layer_size'] = args['layer_size']
    params['learning_rate'] = args['learning_rate']
    params['batch_size'] = args['batch_size']
    params['dropout_keep_probability'] = args['dropout_keep_probability']
    params['validation_window'] = args['validation_window']

    trows = csv_reader.read_csv_dataframe(config.get_rel_path("PATHS", "training_file"))
    vrows = csv_reader.read_csv_dataframe(config.get_rel_path("PATHS", "validation_file"))

    with tf.Graph().as_default():
        loss = run_MLP(params, trows, vrows)

    with tf.Graph().as_default():
        loss = run_MLP(params, trows, vrows)

    return loss


trials = hyperopt.Trials()


def optimize():
    space = {
        'l1_reg': hyperopt.hp.choice('l1_reg', [0.002, 0.001, 0.0001]),
        'l2_reg': hyperopt.hp.choice('l2_reg', [0.002, 0.001]),
        'learning_rate': hyperopt.hp.choice('learning_rate', [0.0001, 0.001]),
        'num_layers': hyperopt.hp.choice('num_layers', [1, 2, 3]),
        'layer_size': hyperopt.hp.choice('layer_size', [1164]),
        'batch_size': hyperopt.hp.choice('batch_size', [32, 64, 128]),
        'dropout_keep_probability': hyperopt.hp.choice('dropout_keep_probability', [0.5, 0.4, 0.3]),
        'validation_window': hyperopt.hp.choice('validation_window', [5])

    }

    best_model = hyperopt.fmin(objective, space, algo=hyperopt.tpe.suggest, trials=trials, max_evals=1)

    print(best_model)
    print("*" * 150)
    print(hyperopt.space_eval(space, best_model))
    print("*" * 150)
    f = open("trials.log", "w")
    for i, tr in enumerate(trials.trials):
        trail = tr['misc']['vals']
        for key in trail.keys():
            trail[key] = trail[key][0]
        f.write("Trail no. : %i\n" % i)
        f.write(str(hyperopt.space_eval(space, trail)) + "\n")
        f.write("Loss : " + str(tr['result']['loss']) + "\n")
        f.write("*" * 100 + "\n")
    f.close()


if __name__ == '__main__':
    optimize()
