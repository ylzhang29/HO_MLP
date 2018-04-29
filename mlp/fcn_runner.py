import os
import threading

import numpy as np
import tensorflow as tf
import sklearn

import utils
from mlp.fcn import FCN


class FCNRunner:
    """
    This class acts as a factory and controller for fcn.py
    FullyConnectedNet builds a tensorflow graph that represents the NN and its evaluation ops.
    FCNRunner uses the FullyConnectedNet graph to build two other graphs: one for training and one for validation.
    A good thing is that both the training and testing graphs share the same variables (https://www.tensorflow.org/versions/r0.11/how_tos/variable_scope/index.html)
    So there is no memory duplication of parameters, or duplication of the process of building up the NN twice.
    +----------------------------------------------------------------------------+
    | training                                                                   |
    | data                                                                       |
    | pipeline                                                                   |
    |    +      +----------+                                                     |
    |    +----> | Fully    +-------> train_loss, train_accuracy, optimization_op |
    |           | Connected|                                                     |
    |    +----> | Net      +-------> validation_loss, validation_accuracy        |
    |    +      +----------+                                                     |
    | validation                                                                 |
    | data                                                                       |
    | pipeline                                                                   |
    +----------------------------------------------------------------------------+
    The training output ops (train_loss, etc...) are only concerned with applying the FCN to the training data.
    The validation output ops (validation_loss, etc...) are only concerned with applying the FCN to the validation data.
    """

    def __init__(self, config, params):

        self.config = config

        # config:
        self.log_folder = config.get_rel_path("PATHS", "log_folder")
        self.experiment_ID = config.get("PROCESS", "experiment_ID") or utils.date_time_string()
        self.validation_interval = config.getint("PROCESS", "validation_interval", fallback=15)
        # self.keep_prob = config.getfloat("TRAINING", "dropout_keep_probability", fallback=1.0)
        self.keep_prob = params['dropout_keep_probability']
        self.num_epochs = config.getint("TRAINING", "num_epochs", fallback=0)
        self.batch_size = params['batch_size']

        self.network = FCN(config, params)
        self.validation_window = params['validation_window']
        self.val_check_after = config.getint("PROCESS", "val_check_after", fallback=1000)

    def bind_training_dataqueue_dataframe(self, train_data_cols, params):
        config = self.config

        # train_batch_size = config.getint("TRAINING", "batch_size")
        train_batch_size = params['batch_size']
        with tf.name_scope("Train"):
            self.network.bind_graph_dataframe("TRAIN", train_data_cols,
                                              train_batch_size,
                                              reuse=False,
                                              with_training_op=True)
        self.train_op = self.network.train_op
        self.train_loss = self.network.loss
        self.train_str_accu = self.network.streaming_accu_op
        self.train_accuracy = self.network.accuracy

        self.train_summaries_merged = self.network.get_summaries()

    def bind_validation_dataqueue_dataframe(self, valid_data_cols):
        config = self.config

        # now reuse the graph to bind new OPs that handle the validation data:
        valid_batch_size = config.getint("TRAINING", "validation_batch_size")
        # TODO this reuse story should be also figured out
        with tf.name_scope("Valid"):
            self.network.bind_graph_dataframe("VALID", valid_data_cols, valid_batch_size, reuse=True, with_training_op=False)
        self.valid_loss = self.network.loss
        self.valid_str_accu = self.network.streaming_accu_op
        self.valid_accuracy = self.network.accuracy
        self.valid_auc = self.network.auc

        self.valid_summaries_merged = self.network.get_summaries()

    # TODO fix this
    def bind_test_dataqueue(self, test_data_cols):
        config = self.config

        # now resuse the graph to bind new OPS that handle the test data:
        test_batch_size = config.getint("TEST", "batch_size")
        with tf.name_scope("Test"):
            self.network.bind_graph("TEST", test_data_cols, test_batch_size, reuse=True, with_training_op=False)
        self.test_loss = self.network.loss
        self.test_str_accu = self.network.streaming_accu_op
        self.test_accuracy = self.network.accuracy
        self.test_summaries_merged = self.network.get_summaries()
        self.test_predictions = self.network.predictions
        self.test_pred_path = config.get_rel_path("TEST", "write_predictions_to")

    def initialize(self):
        config = self.config
        self.session = tf.Session()

        # self.saver = tf.train.Saver(tf.global_variables())
        self.checkpoint_every = config.getint("PROCESS", "checkpoint_every")
        self.checkpoint_path = config.get_rel_path("PATHS", "checkpoint_dir") + "/training.ckpt"

        # load_checkpoint = config.get("PROCESS", "initialize_with_checkpoint") or None
        # if load_checkpoint:
        # self.load_checkpoint(load_checkpoint)

        if config.getint("TRAINING", "num_epochs") > 0:
            self.session.run(tf.global_variables_initializer())

        self.session.run(tf.local_variables_initializer())  # for streaming metrics

        # self.create_summary_writers()

        # coord = tf.train.Coordinator()
        # tf.train.start_queue_runners(sess=self.session, coord=coord)
        # start_queue_runners has to be called for any Tensorflow graph that uses queues.

        # tensorboard_thread = threading.Thread(target=self.start_tensorboard, args=())
        # tensorboard_thread.start()

    def create_summary_writers(self):

        if hasattr(self, "train_summaries_merged"):
            self.train_summary_writer = tf.summary.FileWriter("%s/%s_train" % (self.log_folder, self.experiment_ID),
                                                              self.session.graph)

        if hasattr(self, "valid_summaries_merged"):
            self.valid_summary_writer = tf.summary.FileWriter("%s/%s_valid" % (self.log_folder, self.experiment_ID))

        if hasattr(self, "test_summaries_merged"):
            self.test_summary_writer = tf.summary.FileWriter("%s/%s_test" % (self.log_folder, self.experiment_ID))

    def close_session(self):
        self.session.close()

    def test(self, test_features, test_labels):
        pass

    def train_once_dataframe(self, i, input_batch, label_batch):
        _, train_loss, training_summary, training_accuracy, train_streaming_accuracy = self.session.run(
            [self.train_op, self.train_loss, self.train_summaries_merged, self.train_accuracy, self.train_str_accu],
            feed_dict={self.network.keep_prob: self.keep_prob,
                       self.network.is_training: True,
                       self.network.input_features_placeholder: input_batch,
                       self.network.input_label_placeholder: label_batch})

        # self.train_summary_writer.add_summary(training_summary, i)

        # TODO: fix the printing
        print("Training at the end of iteration %i:\tAccuracy:\t%f\tStreaming Accu:\t%f\tloss:\t%f" % (
            i, training_accuracy, train_streaming_accuracy, train_loss))
        # self.train_summary_writer.flush()
        return train_loss

    def load_checkpoint(self, path):
        self.saver.restore(self.session, path)
        print("Checkpoint loaded from %s" % path)

    def validate_once(self, i, input_batch, label_batch):
        validation_summary, validation_accuracy, validation_streaming_accuracy, validation_loss = self.session.run(
            [self.valid_summaries_merged, self.valid_accuracy, self.valid_str_accu, self.valid_loss],
            feed_dict={self.network.keep_prob: 1,
                       self.network.is_training: False,
                       self.network.input_features_placeholder: input_batch,
                       self.network.input_label_placeholder: label_batch})
        val_auc = self.session.run(
            [self.valid_auc], feed_dict={self.network.keep_prob: 1,
                                         self.network.is_training: False,
                                         self.network.input_features_placeholder: input_batch,
                                         self.network.input_label_placeholder: label_batch})

        val_auc = val_auc[0][1]

        # self.valid_summary_writer.add_summary(validation_summary, i)

        print("\n\n" + "*" * 80)
        print("Validation after iteration %i:\tAccuracy:\t%f\tStreaming Accu:\t%f\tloss:\t%f\tAUC:\t%f" % (
            i, validation_accuracy, validation_streaming_accuracy, validation_loss, val_auc))
        print("*" * 80 + "\n\n")
        # self.valid_summary_writer.flush()
        return val_auc, validation_loss

    def test_once(self):
        test_summary, test_loss, test_predictions, test_accuracy = self.session.run(
            [self.test_summaries_merged, self.test_loss, self.test_predictions, self.test_accuracy],
            feed_dict={self.network.keep_prob: 1, self.network.is_training: False})

        # self.test_summary_writer.add_summary(test_summary, 1)

        print("\n\n" + "*" * 80)
        print("Test accuracy at the end:\t%f\tloss:\t%f" % (
            test_accuracy, test_loss))
        print("*" * 80 + "\n\n")
        # self.test_summary_writer.flush()

        np.savetxt(self.test_pred_path, test_predictions, '%.7f')
        print("Test predictions/scores saved in %s " % self.test_pred_path)

    def start_tensorboard(self):
        log_dir_abs_path = os.path.abspath(self.log_folder)
        print("tensorboard --logdir=%s\n" % (log_dir_abs_path))
        # Popen(["tensorboard", "--logdir=%s" %(log_dir_abs_path)])
        # print("\n")

        utils.background_process(["tensorboard", "--logdir=%s" % (log_dir_abs_path)])

    def run_training_dataframe(self, train_df, validate_df):

        training_size = train_df.shape[0]
        number_of_train_batches = training_size // self.batch_size + 1

        validate_size = validate_df.shape[0]
        number_of_validate_batches = validate_size // self.batch_size + 1

        self.newest_checkpoint_path = ""
        self.last_train_iteration = 0

        print("\n")
        val_acc = []
        avg_validation_acc = []
        val_loss = []
        avg_validation_loss = []
        v_count = 0
        # validation_window = params['validation_window']
        j = 0
        for i in range(1, self.num_epochs + 1):

            train_df = sklearn.utils.shuffle(train_df)

            for batch in range(number_of_train_batches):
                begin_batch = batch * self.batch_size
                batch_slice = slice(begin_batch, min(training_size, begin_batch + self.batch_size))
                # TODO fix the slicing
                input_batch = train_df.iloc[batch_slice, 1:-1]
                label_batch = train_df.iloc[batch_slice, -1]
                train_loss = self.train_once_dataframe(j, input_batch, label_batch)
                self.last_train_iteration = j
                j += 1

            if i % self.validation_interval == 0:

                # TODO understand the validation part (important)
                # TODO fix the slicing
                input_batch = validate_df.iloc[:, 1:-1]
                label_batch = validate_df.iloc[:, -1]

                accuracy, loss = self.validate_once(i, input_batch, label_batch)
                val_acc.append(accuracy)
                val_loss.append(loss)
                v_count += 1
                if v_count > self.validation_window:
                    Validation_Acc = np.mean(val_acc[-self.validation_window:])
                    avg_validation_acc.append(Validation_Acc)
                    avg_validation_loss.append(np.mean(val_loss[-self.validation_window:]))
                else:
                    Validation_Acc = np.mean(val_acc)
                    avg_validation_acc.append(Validation_Acc)
                    avg_validation_loss.append(np.mean(val_loss))

            if j % self.val_check_after == 0:
                if np.mean(avg_validation_loss[:len(avg_validation_loss) // 2]) < np.mean(avg_validation_loss[len(avg_validation_loss) // 2:]):
                    print(np.mean(avg_validation_loss[:len(avg_validation_loss) // 2]))
                    print(np.mean(avg_validation_loss[len(avg_validation_loss) // 2:]))
                    print(self.num_epochs)
                    print("_"*50)
                    break
                else:
                    avg_validation_acc = []
                    avg_validation_loss = []

            # if j % self.val_check_after == 0:
            #     if np.mean(avg_validation_acc[:len(avg_validation_acc) // 2]) < np.mean(avg_validation_acc[len(avg_validation_acc) // 2:]):
            #         print(np.mean(avg_validation_acc[:len(avg_validation_acc) // 2]))
            #         print(np.mean(avg_validation_acc[len(avg_validation_acc) // 2:]))
            #         print(self.num_epochs)
            #         print("_"*50)
            #         break
            #     else:
            #         avg_validation_acc = []
        return Validation_Acc

    def run_test(self):
        print("TESTING")
        self.test_once()
