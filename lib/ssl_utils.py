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

"""Utilities for doing semisupervised learning."""
import tensorflow as tf


def entropy_from_logits(logits):
    """Computes entropy from classifier logits.

    Args:
        logits: a tensor of shape (batch_size, class_count) representing the
        logits of a classifier.

    Returns:
        A tensor of shape (batch_size,) of floats giving the entropies
        batchwise.
    """
    distribution = tf.contrib.distributions.Categorical(logits=logits)
    return distribution.entropy()


def entropy_penalty(logits, entropy_penalty_multiplier, mask):
    """Computes an entropy penalty using the classifier logits.

    Args:
        logits: a tensor of shape (batch_size, class_count) representing the
            logits of a classifier.
        entropy_penalty_multiplier: A float by which the entropy is multiplied.
        mask: A tensor that optionally masks out some of the costs.

    Returns:
        The mean entropy penalty
    """
    entropy = entropy_from_logits(logits)
    losses = entropy * entropy_penalty_multiplier
    losses *= tf.cast(mask, tf.float32)
    return tf.reduce_mean(losses)


def kl_divergence_from_logits(logits_a, logits_b):
    """Gets KL divergence from logits parameterizing categorical distributions.

    Args:
        logits_a: A tensor of logits parameterizing the first distribution.
        logits_b: A tensor of logits parameterizing the second distribution.

    Returns:
        The (batch_size,) shaped tensor of KL divergences.
    """
    distribution1 = tf.contrib.distributions.Categorical(logits=logits_a)
    distribution2 = tf.contrib.distributions.Categorical(logits=logits_b)
    return tf.contrib.distributions.kl_divergence(distribution1, distribution2)


def mse_from_logits(output_logits, target_logits):
    """Computes MSE between predictions associated with logits.

    Args:
        output_logits: A tensor of logits from the primary model.
        target_logits: A tensor of logits from the secondary model.

    Returns:
        The mean MSE
    """
    diffs = tf.nn.softmax(output_logits) - tf.nn.softmax(target_logits)
    squared_diffs = tf.square(diffs)
    return tf.reduce_mean(squared_diffs, -1)


def diff_costs(mode, diff_mask, output_logits, target_logits, diff_multiplier):
    """Computes diff costs given logits.

    Args:
        output_logits: A tensor of logits from the primary model.
        target_logits: A tensor of logits from the secondary model.
        diff_multiplier : A scalar multiplier for the cost.
        diff_mask: A tensor that optionally masks out some of the costs.
        mode: A string controlling the specific cost that is used.

    Returns:
        The mean cost

    Raises:
        ValueError: if the mode is not supported.
    """
    if mode == "forward_kl":
        losses = kl_divergence_from_logits(output_logits, target_logits)
    elif mode == "reverse_kl":
        losses = kl_divergence_from_logits(target_logits, output_logits)
    elif mode == "mean_squared_error":
        losses = mse_from_logits(output_logits, target_logits)
    else:
        raise ValueError("Unsupported mode: {}".format(mode))

    losses *= diff_multiplier
    losses *= tf.cast(diff_mask, tf.float32)
    return tf.reduce_mean(losses)
