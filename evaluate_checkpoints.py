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

"""Evaluate the model.

This script will loop until it has seen the final checkpoint,
waiting for new checkpoints and evaluating the model
on the test set or the validation set using those new checkpoints.
"""

from __future__ import division
import random
import sys
from absl import app
from absl import flags
import numpy as np
import tensorflow as tf
from absl import logging
from lib import dataset_utils
from lib import hparams
from lib.ssl_framework import SSLFramework
from lib import networks


# Flags for model training
# Evalutation binary never needs to know secondary dataset name or
# the mixing factor.
flags.DEFINE_string(
    "hparam_string", None, "String from which we parse hparams."
)
flags.DEFINE_string(
    "primary_dataset_name", "svhn", "Name of dataset containing primary data."
)
flags.DEFINE_string("split", "test", "train or test or valid.")
flags.DEFINE_integer("examples_to_take", 100, "Number of examples to use.")
flags.DEFINE_integer("num_evals", 10, "Total number of evals to run")
flags.DEFINE_integer(
    "shuffle_buffer_size", 1000, "Size of the shuffle buffer."
)
flags.DEFINE_integer(
    "training_length", 500000, "number of steps to train for."
)
flags.DEFINE_string(
    "consistency_model", "mean_teacher", "Which consistency model to use."
)
flags.DEFINE_string(
    "zca_input_file_path",
    "",
    "Path to ZCA input statistics. '' means don't ZCA.",
)

flags.DEFINE_string(
    "checkpoints",
    None,
    "A comma delimited list of checkpoint file prefixes to "
    "evaluate and then exit, e.g. "
    '"/dir1/model.ckpt-97231,/dir2/model.ckpt-1232".',
)
flags.mark_flag_as_required("checkpoints")

FLAGS = flags.FLAGS

# Dummy value that is more likely to alert us to bugs that might arise
# from looking at the value of n_labeled during eval.
_EVAL_N_LABELED = -4000000000


def evaluate(hparams):
    """Evalute a set of checkpoints multiple times."""
    accuracies = {}
    for explicit_checkpoint_path in FLAGS.checkpoints.split(","):
        logging.info(explicit_checkpoint_path)
        accuracies[explicit_checkpoint_path] = []

        tf.reset_default_graph()
        coord = tf.train.Coordinator()
        with tf.device("/cpu:0"):
            images, labels = make_images_and_labels_tensors()

            # Construct model to register global variables.
            ssl_framework = SSLFramework(
                networks.wide_resnet,
                hparams,
                images,
                labels,
                make_train_tensors=False,
                consistency_model=FLAGS.consistency_model,
                zca_input_file_path=FLAGS.zca_input_file_path,
            )
        new_saver = tf.train.Saver(tf.global_variables())

        with tf.Session() as sess:
            new_saver.restore(sess, explicit_checkpoint_path)

            # For some reason, we need to initialize the tables again
            sess.run(tf.tables_initializer())

            for _ in range(FLAGS.num_evals):
                # Start the input enqueue threads
                tf.train.start_queue_runners(sess=sess, coord=coord)

                # Evaluate the model
                all_images, all_labels = sess.run([images, labels])

                feed_dict = {
                    ssl_framework.inputs: all_images,
                    ssl_framework.is_training: False,
                }
                output = sess.run(ssl_framework.logits, feed_dict=feed_dict)

                correct = np.sum(np.argmax(output, 1) == all_labels)
                accuracy = float(correct) / float(FLAGS.examples_to_take)

                logging.info("Accuracy: %f", accuracy)
                accuracies[explicit_checkpoint_path].append(accuracy)

    logging.info(accuracies)


def make_images_and_labels_tensors():
    """Make tensors for loading images and labels from dataset."""

    with tf.name_scope("input"):
        dataset = dataset_utils.get_dataset(
            FLAGS.primary_dataset_name, FLAGS.split
        )
        # Shuffle with the same seed to allow comparisons between models on the same subsampled validation sets.
        dataset = dataset.shuffle(FLAGS.shuffle_buffer_size, seed=0)
        dataset = dataset.batch(FLAGS.examples_to_take)

        # Get the actual results from the iterator
        iterator = dataset.make_initializable_iterator()
        tf.add_to_collection(
            tf.GraphKeys.TABLE_INITIALIZERS, iterator.initializer
        )
        images, labels, _ = iterator.get_next()
        images = tf.cast(images, tf.float32)

    return images, labels


def main(_):
    hps = hparams.get_hparams(
        FLAGS.primary_dataset_name, FLAGS.consistency_model
    )
    if FLAGS.hparam_string:
        hps.parse(FLAGS.hparam_string)
    evaluate(hps)


if __name__ == "__main__":
    app.run(main)
