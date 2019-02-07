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
import json
import os
import time
from absl import app
from absl import flags
import numpy as np
import tensorflow as tf
from absl import logging
from lib import dataset_utils
from lib import tf_utils
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
flags.DEFINE_string("split", "test", "test or valid.")
flags.DEFINE_integer("batch_size", 100, "Size of the batch.")
flags.DEFINE_integer("examples_to_take", -1, "Number of examples to use.")
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
    "labeled_classes_filter",
    "",
    "Comma-delimited list of class numbers from labeled "
    "dataset to use during training. Defaults to all classes.",
)

# Flags for book-keeping
flags.DEFINE_string(
    "root_dir", None, "The overall dir in which we store experiments"
)
flags.mark_flag_as_required("root_dir")

flags.DEFINE_string(
    "experiment_name", "default", "The name of this particular experiment"
)
flags.DEFINE_string(
    "evaluate_single_checkpoint",
    "",
    "If set, defines the checkpoint file prefix to evaluate "
    'and then exit, e.g. "model.ckpt-97231".',
)

FLAGS = flags.FLAGS

# Dummy value that is more likely to alert us to bugs that might arise
# from looking at the value of n_labeled during eval.
_EVAL_N_LABELED = -4000000000
TOP_K_VALUE = 5


def top_k(logits, k):
    """Get the indices of the top-k logits."""
    return np.argpartition(logits, -k, axis=1)[:, -k:]


def evaluate(hps, result_dir, tuner=None, trial_name=None):
    """Run an eval loop for particular hyperparameters."""

    container_name = trial_name or ""
    with tf.container(container_name):

        logging.info(
            "Evaluating model using dataset %s", FLAGS.primary_dataset_name
        )

        # We shouldn't start trying to read files until the directory exists
        while not tf.gfile.Exists(result_dir):
            logging.info(
                "Waiting on result directory creation: %s", result_dir
            )
            time.sleep(60)

        # Once we know result_dir exists, it's safe to create summary writer.
        # This writer persists across all sessions and graphs
        summary_dir = os.path.join(result_dir, FLAGS.split)
        if not tf.gfile.Exists(summary_dir):
            tf.gfile.MakeDirs(summary_dir)
        writer = tf.summary.FileWriter(summary_dir)

        # Keep track of which files we've processed
        # If this process gets pre-empted, we may re-process some files.
        # I won't fix this unless it becomes a performance issue
        processed_files = []

        # Keep track of whether we've seen the last step
        seen_last_step = False

        while True:  # Loop forever
            if tuner and tuner.should_trial_stop():
                logging.info(
                    "Got tuner.should_trial_stop(). Calling tuner.report_done()."
                )
                # Signal to the tuner that the trial is done.
                tuner.report_done()
                return

            images, labels = make_images_and_labels_tensors(
                FLAGS.examples_to_take
            )

            if FLAGS.evaluate_single_checkpoint:
                meta_file = FLAGS.evaluate_single_checkpoint + ".meta"
            else:

                potential_files = tf.gfile.ListDirectory(result_dir)
                meta_files = [
                    f for f in potential_files if f.endswith(".meta")
                ]
                prefix = "model.ckpt-"
                meta_files = sorted(
                    meta_files,
                    key=lambda s: int(os.path.splitext(s)[0][len(prefix) :]),
                )
                meta_files = [
                    f for f in meta_files if f not in processed_files
                ]

                if meta_files:
                    meta_file = meta_files[0]
                else:
                    if seen_last_step:
                        logging.info(
                            "No new checkpoints and last step seen. FINISHED."
                        )
                        if tuner:
                            logging.info("Calling tuner.report_done().")
                            # Signal to the tuner that the trial is done.
                            tuner.report_done()
                        break
                    else:
                        logging.info(
                            "No new checkpoints, sleeping for 60 seconds"
                        )
                        time.sleep(60)
                        continue

            # Load the model
            explicit_meta_path = os.path.join(result_dir, meta_file)
            explicit_checkpoint_path = explicit_meta_path[: -len(".meta")]

            # We need a coordinator for threads and stuff
            coord = tf.train.Coordinator()

            # Construct model
            ssl_framework = SSLFramework(
                networks.wide_resnet,
                hps,
                images,
                labels,
                make_train_tensors=False,
                consistency_model=FLAGS.consistency_model,
                zca_input_file_path=FLAGS.zca_input_file_path,
            )
            new_saver = tf.train.Saver(tf.global_variables())

            # We need a new session for each checkpoint
            sess = tf.Session()
            logging.info("New eval session created.")

            # Start the input enqueue threads
            tf.train.start_queue_runners(sess=sess, coord=coord)

            new_saver.restore(sess, explicit_checkpoint_path)

            # For some reason, we need to initialize the tables again
            sess.run(tf.tables_initializer())

            # Evaluate the model
            correct = 0
            top_k_correct = 0
            total_seen = 0

            logging.info("Evaluating batches.")
            while True:
                fetches = [ssl_framework.global_step, ssl_framework.logits, labels]
                feed_dict = {
                    ssl_framework.is_training: False,
                }
                try:
                    step, output, label_batch = sess.run(fetches, feed_dict=feed_dict)
                except tf.errors.OutOfRangeError:
                    break

                correct += np.sum(np.argmax(output, 1) == label_batch)
                this_top_k = top_k(output, TOP_K_VALUE)
                top_k_correct += (
                    (
                        np.reshape(label_batch, (label_batch.shape[0], 1))
                        - this_top_k
                    )
                    == 0
                ).sum()

                total_seen += output.shape[0]

            sess.close()

            # We need to reset the graph
            tf.reset_default_graph()

            accuracy = float(correct) / float(total_seen)
            top_k_accuracy = float(top_k_correct) / float(total_seen)
            result_dict = {
                "step": str(step),
                "accuracy": str(accuracy),
                "top_k_accuracy": str(top_k_accuracy),
            }
            logging.info(str(result_dict))

            if FLAGS.evaluate_single_checkpoint:
                return

            result_filename = "evaluation_results_{}_{}".format(
                FLAGS.split, str(step)
            )
            path = os.path.join(result_dir, result_filename)

            # Actually dump json out to file
            with tf.gfile.FastGFile(path, "w") as f:
                json.dump(result_dict, f)

            # Write the summaries
            for label, value in zip(
                ["accuracy", "top_k_accuracy"], [accuracy, top_k_accuracy]
            ):
                tag = "eval/" + FLAGS.split + "_" + label
                summary = tf.Summary(
                    value=[tf.Summary.Value(tag=tag, simple_value=value)]
                )
                writer.add_summary(summary, global_step=step)
            writer.flush()

            # Report the eval stats to the tuner.
            if tuner:
                metrics = {
                    "accuracy": accuracy,
                    "top_k_accuracy": top_k_accuracy,
                }

                should_stop = tuner.report_measure(
                    accuracy, metrics=metrics, global_step=step
                )

                if should_stop:
                    logging.info(
                        "Got should_stop. Calling tuner.report_done()."
                    )
                    # Signal to the tuner that the trial is done.
                    tuner.report_done()

            # We will only ever call a file processed if the data has been written.
            # Thus, we should process all the files at least once.
            processed_files.append(meta_file)

            if step >= FLAGS.training_length:
                seen_last_step = True


def make_images_and_labels_tensors(examples_to_take):
    """Make tensors for loading images and labels from dataset."""

    with tf.name_scope("input"):
        dataset = dataset_utils.get_dataset(
            FLAGS.primary_dataset_name, FLAGS.split
        )
        dataset = dataset.filter(make_labeled_data_filter())

        # This is necessary for datasets that aren't shuffled on disk, such as
        # ImageNet.
        if FLAGS.split == "train":
            dataset = dataset.shuffle(FLAGS.shuffle_buffer_size, 0)

        # Optionally only use a certain fraction of the dataset.
        # This is used in at least 2 contexts:
        # 1. We don't evaluate on all training data sometimes for speed reasons.
        # 2. We may want smaller validation sets to see whether HPO still works.
        if examples_to_take != -1:
            dataset = dataset.take(examples_to_take)

        # Batch the results
        dataset = dataset.batch(FLAGS.batch_size)

        # Get the actual results from the iterator
        iterator = dataset.make_initializable_iterator()
        tf.add_to_collection(
            tf.GraphKeys.TABLE_INITIALIZERS, iterator.initializer
        )
        images, labels, _ = iterator.get_next()
        images = tf.cast(images, tf.float32)

    return images, labels


def make_labeled_data_filter():
    """Make filter for certain classes of labeled data."""
    class_filter = tf_utils.filter_fn_from_comma_delimited(
        FLAGS.labeled_classes_filter
    )
    return lambda image, label, fkey: class_filter(label)


def main(_):
    result_dir = os.path.join(FLAGS.root_dir, FLAGS.experiment_name)
    hps = hparams.get_hparams(
        FLAGS.primary_dataset_name, FLAGS.consistency_model
    )
    if FLAGS.hparam_string:
        hps.parse(FLAGS.hparam_string)
    evaluate(hps, result_dir)


if __name__ == "__main__":
    app.run(main)
