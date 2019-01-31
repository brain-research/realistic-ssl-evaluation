#!/usr/bin/python3
#
# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Hyperparameter class for Pi-model and Mean Teacher models.

This file contains functions to create default hyperparameters
for various datasets and configuration for training semisupervised
learning models with the pi-model or mean teacher techniques.
"""

from __future__ import absolute_import
from __future__ import division

import tensorflow as tf

base = dict(
    warmup_steps=200000,
    initial_lr=3e-3,
    horizontal_flip=False,
    random_translation=True,
    gaussian_noise=True,
    consistency_func="forward_kl",
    max_cons_multiplier=1.0,
    entropy_penalty_multiplier=0.,
    ema_factor=0.95,
    vat_epsilon=6.0,  # Norm length of perturbation
    vat_xi=1e-6,  # Small constant for computing the finite difference in VAT
    lr_decay_steps=400000,
    lr_decay_rate=0.2,
    num_classes=10,
    width=2,
    num_residual_units=4,
    lrelu_leakiness=0.1,
    pseudo_label_threshold=0.95,
)


def merge_dicts(x, y):
    z = x.copy()
    z.update(y)
    return z


# HParam overrides for different consistency functions
consistency_model_overrides = dict(
    mean_teacher=dict(
        consistency_func="mean_squared_error",
        max_cons_multiplier=8.0,
        initial_lr=4e-4,
    ),
    vat=dict(consistency_func="forward_kl", max_cons_multiplier=0.3),
    pi_model=dict(
        consistency_func="mean_squared_error",
        max_cons_multiplier=20.0,
        initial_lr=3e-4,
    ),
    pseudo_label=dict(
        consistency_func="reverse_kl", max_cons_multiplier=1.0, initial_lr=3e-4
    ),
    none=dict(max_cons_multiplier=0.),
)

# HParam overrides for different datasets
cifar10_overrides = dict(horizontal_flip=True)
cifar_unnormalized_overrides = cifar10_overrides

imagenet_overrides = dict(
    horizontal_flip=True,
    num_classes=1000,
    random_translation=False,
    gaussian_noise=False,
)
svhn_overrides = dict(gaussian_noise=False, vat_epsilon=1.0)
dataset_overrides = dict(
    cifar10=cifar10_overrides,
    cifar_unnormalized=cifar_unnormalized_overrides,
    imagenet=imagenet_overrides,
    imagenet_32=imagenet_overrides,
    imagenet_64=imagenet_overrides,
    svhn=svhn_overrides,
)


def get_hparams(dataset, consistency_model):
    return tf.contrib.training.HParams(
        **merge_dicts(
            merge_dicts(base, dataset_overrides[dataset]),
            consistency_model_overrides[consistency_model],
        )
    )
