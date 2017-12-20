#!/usr/bin/env python
import torch
import numpy as np
import torch.nn as nn
import DocumentClassifier_cnn as doc_cnn
import DocumentClassifier_cnn_hier as doc_cnn_hier
import sys
import os
sys.path.insert(0,
                os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'model'))


def xavier_init(layer):
    size = layer.weight.size()
    fan_out = size[0]  # number of rows
    fan_in = size[1]  # number of columns
    variance = 1. / np.sqrt(fan_in / 2.)
    layer.weight = nn.Parameter(torch.randn((fan_out, fan_in))*variance,
                                requires_grad=True)
    layer.bias = nn.Parameter(torch.zeros(fan_out), requires_grad=True)


def main(model_config):
    if model_config["model_type"] == 'DocumentClassifier_cnn':
        model = doc_cnn(model_config)
    elif model_config["model_type"] == 'DocumentClassifier_cnn_hier':
        model = doc_cnn_hier(model_config)

    if model_config["mode"] == 'train':
        model.train()
    elif model_config["mode"] == 'test':
        model.eval()
    else:
        raise Exception()

    if model_config["initial_train"]:
        return model
    else:  # load parameters
        state_dict = torch.load(os.path.join(model_config["current_dir"],
                                             model_config["param_dir"],
                                             model_config["model_type"]))
        own_state = model.state_dict()
        for name, param in state_dict.items():
            own_state[name].copy_(param)

    return model

