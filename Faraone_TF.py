import os
import tensorflow as tf
import config_reader
import csv_reader
import mlp
import utils
import sys,os

def run_MLP(params, trows, vrows):

    config = config_reader.read_config(utils.abs_path_of("config/default.ini"))

    if not os.path.isdir(config.get_rel_path("PATHS", "checkpoint_dir")):
        utils.mkdir_recursive(config.get_rel_path("PATHS", "checkpoint_dir"))

    iris_runner = mlp.FCNRunner(config, params)  # trows, vrows, test_rows, config)

    iris_runner.bind_training_dataqueue_dataframe(trows,params)
    iris_runner.bind_validation_dataqueue_dataframe(vrows)

    # TODO: fix the bind_test_dataqueue
    '''
    if "TEST" in config:
        test_path = config.get_rel_path("TEST","test_file")
        with tf.name_scope("test_data"):
            test_rows = csv_reader.read_test_csv(test_path, int(config["TEST"]["batch_size"]))
        iris_runner.bind_test_dataqueue(test_rows)
    '''
    iris_runner.initialize()

    if "TRAINING" in config:
        valid_acc = iris_runner.run_training_dataframe(trows, vrows)
        iris_runner.close_session()
    #if "TEST" in config:
        #iris_runner.run_test()

    return 1 - valid_acc
