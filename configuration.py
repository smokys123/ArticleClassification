#!/usr/bin/env python
import os
import json


class Config(object):
    def __init__(self, config_dict):
        self.data_dir = config_dict["data_dir"]
        self.meta_dir = config_dict["meta_dir"]
        self.param_dir = config_dict["param_dir"]
        self.out_dir = config_dict["out_dir"]
        self.fine_tune = bool(config_dict["fine_tune"])
        self.use_word2vec_for_unk = bool(config_dict["use_word2vec_for_unk"])
        self.lower_word = bool(config_dict["lower_word"])
        self.use_batch = bool(config_dict["use_batch"])
        self.train_epoch = int(config_dict["train_epoch"])
        self.model_save_period = int(config_dict["model_save_period"])
        self.batch_size = int(config_dict["batch_size"])
        self.learning_rate = float(config_dict["learning_rate"])
        self.initial_train = bool(config_dict["initial_train"])
        self.param_init_method = config_dict["param_init_method"]
        self.mode = config_dict["mode"]
        self.batch_shuffle = bool(config_dict["batch_shuffle"])
        self.shuffle = bool(config_dict["shuffle"])
        self.remove_stopwords = bool(config_dict["remove_stopwords"])
        self.class_level = config_dict["class_level"]
        self.data_type = config_dict["data_type"]  # Yahoo.answers
        self.model_type = config_dict["model_type"]
        self.use_sample_data = config_dict["use_sample_data"]
        self.model_config_dir = config_dict["model_config_dir"]

    def show_config(self):

        print("""
                data_dir				: {}
                data_type				: {}
                meta_dir				: {}
                param_dir				: {}
                out_dir					: {}
                model_config_dir		: {}

                fine_tune				: {}
                initial_train			: {}
                parameter initiation	: {}

                use_word2vec_for_unk 	: {}
                lower_word				: {}
                remove_stopwords		: {}
                class_level				: {}

                use_batch				: {}
                batch_size				: {}
                batch_shuffle			: {}
                shuffle 				: {}

                train_epoch				: {}
                learning_rate			: {}
                mode					: {}
                model_type				: {}
                use_sample_data			: {}
              """.format(self.data_dir, self.data_type, self.meta_dir,
                         self.param_dir, self.out_dir, self.model_config_dir,
                         self.fine_tune, self.initial_train,
                         self.param_init_method, self.use_word2vec_for_unk,
                         self.lower_word, self.remove_stopwords,
                         self.class_level, self.use_batch, self.batch_size,
                         self. batch_shuffle, self.shuffle, self.train_epoch,
                         self.learning_rate, self.mode, self.model_type,
                         self.use_sample_data))


def get_config(args):
    # configuration file to json
    """
    : type -- train or test
    : model_type
    """
    config_dict = {"mode": args.mode}
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'configuration',
                           'config.json'), 'r') as file:
        config_dict.update(json.load(file))
    if args.epoch:
        config_dict["train_epoch"] = int(args.epoch)
    if args.model_type:
        config_dict["model_type"] = args.model_type
    return Config(config_dict)
