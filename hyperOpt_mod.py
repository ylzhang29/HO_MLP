import config_reader
import utils
from Faraone_TF import run_MLP
import hyperopt
import tensorflow as tf
import re
import pickle
import csv_reader


def objective(args):

    params = {}

    params['l1_reg'] = args['l1_reg']
    params['l2_reg'] = args['l2_reg']
    params['num_layers'] = int(args['num_layers'])
    params['layer_size'] = int(args['layer_size'])
    params['learning_rate'] = args['learning_rate']
    params['batch_size'] = args['batch_size']
    params['dropout_keep_probability'] = args['dropout_keep_probability']
    params['validation_window'] = args['validation_window']

    with tf.Graph().as_default():
        loss = run_MLP(params, trows, vrows)

    return loss


# trials = hyperopt.Trials()
# trials = pickle.load(open("trial_obj.pkl", "rb"))

def optimize():
    save_trial = 1
    max_trials = 1

    #   space = {
    #        'l1_reg': hyperopt.hp.uniform('l1_reg', 0, 0.2),
    #        'l2_reg': hyperopt.hp.uniform('l2_reg', 0, 0.2),
    #        'learning_rate': hyperopt.hp.uniform('learning_rate', 0.0000001, 0.0001),
    #        'num_layers': hyperopt.hp.choice('num_layers', [1]),
    #        'layer_size': hyperopt.hp.quniform('layer_size', 10, 50, 5),
    #        'batch_size': hyperopt.hp.choice('batch_size', [8, 16, 32, 64, 128, 256, 512, 1024, 1500, 2000]),
    #        'dropout_keep_probability': hyperopt.hp.uniform('dropout_keep_probability', 0.1, 1),
    #        'validation_window': hyperopt.hp.choice('validation_window',[10])
    #   }

    space = {
        'l1_reg': hyperopt.hp.choice('l1_reg', [0.1335267529016616]),
        'l2_reg': hyperopt.hp.choice('l2_reg', [0.11497692967392144]),
        'learning_rate': hyperopt.hp.choice('learning_rate', [0.000005280539672336577]),
        'num_layers': hyperopt.hp.choice('num_layers', [1]),
        'layer_size': hyperopt.hp.choice('layer_size', [10]),
        'batch_size': hyperopt.hp.choice('batch_size', [512]),
        'dropout_keep_probability': hyperopt.hp.choice('dropout_keep_probability', [0.8846418566190806]),
        'validation_window': hyperopt.hp.choice('validation_window', [10])
    }

    try:
        trials = pickle.load(open("trial_obj.pkl", "rb"))
        print("________Loading saved trials object__________")
        max_trials = len(trials.trials) + save_trial
        print("Rerunning from {} trials to {} (+{}) trials".format(len(trials.trials), max_trials, save_trial))
    except:
        trials = hyperopt.Trials()

    best_model = hyperopt.fmin(objective, space, algo=hyperopt.tpe.suggest, trials=trials, max_evals=max_trials)

    with open("trial_obj.pkl", "wb") as f:
        pickle.dump(trials, f)

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


def main():
    global trows, vrows

    config = config_reader.read_config(utils.abs_path_of("config/default.ini"))

    trows = csv_reader.read_csv_dataframe(config.get_rel_path("PATHS", "training_file"))
    vrows = csv_reader.read_csv_dataframe(config.get_rel_path("PATHS", "validation_file"))

    while True:
        optimize()

if __name__ == '__main__':
    vrows = trows = ""
    main()

