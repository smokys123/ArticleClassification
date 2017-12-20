#!/usr/bin/env python
import argparse
import os
import pandas as pd
from configuration import get_config
from train import train
from evaluation import test


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-model_type",
        help="DocumentClassifier_cnn, DocumentClassifier_cnn_hier",
        required=True,
        choices=['DocumentClassifier_cnn', 'DocumentClassifier_cnn_hier'])
    parser.add_argument(
        "-mode",
        help="train for training, test for testing",
        required=True, choices=['train', 'test'])
    parser.add_argument("-epoch", type=int, help='training epoch',
                        requirements=True)
    return parser.parse_args()


def main():
    args = arg_parse()
    config = get_config(args)
    config.show_config()

    dataset = pd.read_csv(
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     'dataset', 'BBC', 'preproc_dataset.csv'))

    if config.mode == 'train':
        train(config, dataset)
    elif config.mode == 'test':
        test(config, dataset)


if __name__ == '__main__':
    main()
