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

"""Train the model."""

from __future__ import division
import os
from absl import app
from absl import flags
import numpy as np
import tensorflow as tf
from absl import logging
from lib import data_provider
from lib import dataset_utils
from lib import tf_utils
from lib import hparams
from lib.ssl_framework import SSLFramework
from lib import networks


_PRINT_SPAN = 300
_CHECK_TRIAL_EARLY_STOP = 100

# Flags for model training
flags.DEFINE_string(
    "hparam_string", None, "String from which we parse hparams."
)
flags.DEFINE_string(
    "primary_dataset_name", "svhn", "Name of dataset containing primary data."
)
flags.DEFINE_string(
    "secondary_dataset_name",
    "",
    "Name of dataset containing secondary data. Defaults to primary dataset",
)
flags.DEFINE_integer("label_map_index", 0, "Index of the label map.")
flags.DEFINE_integer(
    "n_labeled", -1, "Number of labeled examples, or -1 for entire dataset."
)
flags.DEFINE_integer(
    "training_length", 500000, "number of steps to train for."
)
flags.DEFINE_integer("batch_size", 100, "Size of the batch")
flags.DEFINE_string(
    "consistency_model", "mean_teacher", "Which consistency model to use."
)
flags.DEFINE_string(
    "zca_input_file_path",
    "",
    "Path to ZCA input statistics. '' means don't ZCA.",
)

flags.DEFINE_float(
    "unlabeled_data_random_fraction",
    1.0,
    "The fraction of unlabeled data to use during training.",
)
flags.DEFINE_string(
    "labeled_classes_filter",
    "",
    "Comma-delimited list of class numbers from labeled "
    "dataset to use during training. Defaults to all classes.",
)
flags.DEFINE_string(
    "unlabeled_classes_filter",
    "",
    "Comma-delimited list of class numbers from unlabeled "
    "dataset to use during training. Useful for labeled "
    "datasets being used as unlabeled data. Defaults to all "
    "classes.",
)
flags.DEFINE_string(
    "dataset_mode",
    "mix",
    "'labeled' - use only labeled data to train the model. "
    "'mix' (default) -  use mixed data to train the model",
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
    "load_checkpoint",
    "",
    "Checkpoint file to start training from (e.g. "
    ".../model.ckpt-354615), or None for random init",
)

FLAGS = flags.FLAGS


def train(hps, result_dir, tuner=None, trial_name=None):
    """Construct model and run main training loop."""
    # Write hyperparameters to text summary
    hparams_dict = hps.values()
    # Create a markdown table from hparams.
    header = "| Key | Value |\n| :--- | :--- |\n"
    keys = sorted(hparams_dict.keys())
    lines = ["| %s | %s |" % (key, str(hparams_dict[key])) for key in keys]
    hparams_table = header + "\n".join(lines) + "\n"

    hparam_summary = tf.summary.text(
        "hparams", tf.constant(hparams_table, name="hparams"), collections=[]
    )

    with tf.Session() as sess:
        writer = tf.summary.FileWriter(result_dir, graph=sess.graph)
        writer.add_summary(hparam_summary.eval())
        writer.close()

    # We need to be able to run on the normal dataset for debugging.
    if FLAGS.n_labeled != -1:
        label_map = "label_map_count_{}_index_{}".format(
            FLAGS.n_labeled, FLAGS.label_map_index
        )
    else:
        label_map = None

    container_name = trial_name or ""

    # Create a separate container for each run so parameters don't stick around
    with tf.container(container_name):

        if label_map:
            label_table = dataset_utils.construct_label_table(
                FLAGS.primary_dataset_name, label_map
            )
        else:
            label_table = None

        labeled_data_filter_fn = make_labeled_data_filter_fn(label_table)
        unlabeled_data_filter_fn = make_unlabeled_data_filter_fn()

        images, labels, _, _, _, _ = data_provider.get_simple_mixed_batch(
            labeled_dataset_name=FLAGS.primary_dataset_name,
            unlabeled_dataset_name=(
                FLAGS.secondary_dataset_name or
                FLAGS.primary_dataset_name),
            split="train",
            batch_size=FLAGS.batch_size,
            shuffle_buffer_size=1000,
            labeled_data_filter_fn=labeled_data_filter_fn,
            unlabeled_data_filter_fn=unlabeled_data_filter_fn,
            mode=FLAGS.dataset_mode,
        )

        logging.info("Training data tensors constructed.")
        # This is necessary because presently svhn data comes as uint8
        images = tf.cast(images, tf.float32)
        ssl_framework = SSLFramework(
            networks.wide_resnet,
            hps,
            images,
            labels,
            make_train_tensors=True,
            consistency_model=FLAGS.consistency_model,
            zca_input_file_path=FLAGS.zca_input_file_path,
        )
        tf.summary.scalar("n_labeled", FLAGS.n_labeled)
        tf.summary.scalar("batch_size", FLAGS.batch_size)

        logging.info("Model instantiated.")
        init_op = tf.global_variables_initializer()

        saver = tf.train.Saver(keep_checkpoint_every_n_hours=1)

        if FLAGS.load_checkpoint:
            vars_to_load = [
                v for v in tf.all_variables() if "logit" not in v.name
            ]
            finetuning_saver = tf.train.Saver(
                keep_checkpoint_every_n_hours=1, var_list=vars_to_load
            )

        def init_fn(_, sess):
            sess.run(init_op)
            if FLAGS.load_checkpoint:
                logging.info(
                    "Fine tuning from checkpoint: %s", FLAGS.load_checkpoint
                )
                finetuning_saver.restore(sess, FLAGS.load_checkpoint)

        scaffold = tf.train.Scaffold(
            saver=saver, init_op=ssl_framework.global_step_init, init_fn=init_fn
        )
        logging.info("Scaffold created.")
        monitored_sess = tf.train.MonitoredTrainingSession(
            scaffold=scaffold,
            checkpoint_dir=result_dir,
            save_checkpoint_secs=300,
            save_summaries_secs=10,
            save_summaries_steps=None,
            config=tf.ConfigProto(
                allow_soft_placement=True, log_device_placement=False
            ),
            max_wait_secs=300,
        )
        logging.info("MonitoredTrainingSession initialized.")
        trainable_params = np.sum(
            [
                np.prod(v.get_shape().as_list())
                for v in tf.trainable_variables()
            ]
        )
        logging.info("Trainable parameters: %s", str(trainable_params))

        def should_stop_early():
            if tuner and tuner.should_trial_stop():
                logging.info(
                    "Got tuner.should_trial_stop(). Stopping trial early."
                )
                return True
            else:
                return False

        with monitored_sess as sess:

            while True:
                _, step, values_to_log = sess.run(
                    [
                        ssl_framework.train_op,
                        ssl_framework.global_step,
                        ssl_framework.scalars_to_log,
                    ],
                    feed_dict={ssl_framework.is_training: True},
                )

                if step % _PRINT_SPAN == 0:
                    logging.info(
                        "step %d: %r",
                        step,
                        dict((k, v) for k, v in values_to_log.items()),
                    )
                if step >= FLAGS.training_length:
                    break
                # Don't call should_stop_early() too frequently
                if step % _CHECK_TRIAL_EARLY_STOP == 0 and should_stop_early():
                    break


def make_labeled_data_filter_fn(label_table):
    """Make filter for certain classes of labeled data."""
    class_filter = tf_utils.filter_fn_from_comma_delimited(
        FLAGS.labeled_classes_filter
    )
    if label_table:
        return lambda _, label, fkey: class_filter(label) & label_table.lookup(
            fkey
        )
    else:
        return lambda _, label, fkey: class_filter(label)


def make_unlabeled_data_filter_fn():
    """Make filter for certain classes and a random fraction of unlabeled
    data."""
    class_filter = tf_utils.filter_fn_from_comma_delimited(
        FLAGS.unlabeled_classes_filter
    )

    def random_frac_filter(fkey):
        return tf_utils.hash_float(fkey) < FLAGS.unlabeled_data_random_fraction

    return lambda _, label, fkey: class_filter(label) & random_frac_filter(
        fkey
    )


def main(_):
    result_dir = os.path.join(FLAGS.root_dir, FLAGS.experiment_name)
    hps = hparams.get_hparams(
        FLAGS.primary_dataset_name, FLAGS.consistency_model
    )
    if FLAGS.hparam_string:
        hps.parse(FLAGS.hparam_string)
    train(hps, result_dir)


if __name__ == "__main__":
    app.run(main)
