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

"""Framework code for different SSL models.

Can be used for predictions and training for:
* Pi-Model
* Mean Teacher
* Virtual Adversarial Training.
* Pseudo-Label
"""

from __future__ import division
import functools
import tensorflow as tf
from absl import logging
from lib import dataset_utils
from lib import ssl_utils
from third_party.vat import vat_utils


class SSLFramework(object):
    def __init__(
        self,
        network,
        hps,
        inputs,
        labels,
        make_train_tensors,
        consistency_model,
        zca_input_file_path=None,
    ):
        """Init the class.

        Args:
            network (callable): Function which builds the graph for the
                network.
            hps (tf.contrib.training.HParams): all the hparams
            images (tensor): training images
            labels (int): training labels
            make_train_tensors (bool): make the tensors needed for training?
            consistency_model (str): which consistency regularization model to
                use.
            zca_input_file_path (str): path to pre-computed ZCA statistics.

        Returns:
            Initialized object.

        Raises:
            ValueError: if consistency_model is not valid.
        """

        logging.info("Building model with HParams %s", hps)
        self.network = network
        self.hps = hps
        self.consistency_model = consistency_model
        self.global_step = tf.train.get_or_create_global_step()

        # We need to wrap these tensors in identity calls so that passing
        # in the test data through a feed_dict will work at eval time
        self.inputs = tf.identity(inputs)
        self.labels = tf.identity(labels)
        self.is_training = tf.placeholder(
            dtype=tf.bool, shape=[], name="is_training"
        )

        if zca_input_file_path:
            logging.info(
                "Normalizing images with stats from: %s", zca_input_file_path
            )
            self.processed_images = self.inputs

            # A two step process that does the same as the
            # cifar_unnormalized -> cifar10 conversion process

            # 1. "De-normalize" back into [0, 255]
            self.processed_images /= 2
            self.processed_images += 0.5
            self.processed_images *= 255.0

            # 2. Apply Global Contrast Normalization and ZCA normalization
            # based on some dataset statistics passed in as a hyperparameter.
            self.processed_images = dataset_utils.tf_gcn(self.processed_images)
            self.processed_images = dataset_utils.zca_normalize(
                self.processed_images, zca_input_file_path
            )
        else:
            self.processed_images = self.inputs

        # logits is always the clean network output, and is what is used to evaluate the model.
        #
        # logits_student is the output that we want to make more similar to logits_teacher.
        # Each SSL method has its own concept of what the student and teacher outputs are, but
        # logits is guaranteed to be equal to either logits_student or logits_teacher.
        self.logits, self.logits_student, self.logits_teacher = (
            self.prediction()
        )

        labeled_mask = tf.not_equal(-1, labels)
        masked_logits = tf.boolean_mask(self.logits, labeled_mask)
        masked_labels = tf.boolean_mask(self.labels, labeled_mask)

        self.accuracy = tf.reduce_mean(
            tf.to_float(
                tf.equal(
                    tf.argmax(masked_logits, axis=-1),
                    tf.to_int64(masked_labels),
                )
            )
        )

        num_total_examples = tf.shape(self.inputs)[0]
        self.labeled_loss = tf.reduce_sum(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=masked_logits, labels=masked_labels
            )
        ) / tf.to_float(num_total_examples)

        self.global_step_init = tf.initialize_variables([self.global_step])

        if make_train_tensors:
            self.make_train_tensors()

    def make_train_tensors(self):
        """Makes tensors needed for training this model.

        These shouldn't be made when just doing evaluation, both because it
        wastes memory and because some of these depend on hyperparameters that
        are unavailable during evaluation.

        Raises:
            ValueError: If given invalid hparams.
        """
        self.lr = tf.train.exponential_decay(
            self.hps.initial_lr,
            self.global_step,
            self.hps.lr_decay_steps,
            self.hps.lr_decay_rate,
            staircase=True,
        )

        # Multiplier warm-up schedule from Appendix B.1 of
        # the Mean Teacher paper (https://arxiv.org/abs/1703.01780)
        # "The consistency cost coefficient and the learning rate were ramped up
        # from 0 to their maximum values, using a sigmoid-shaped function
        # e^{−5(1−x)^2}, where x in [0, 1]."
        cons_multiplier = tf.cond(
            self.global_step < self.hps.warmup_steps,
            lambda: tf.exp(
                -5.0
                * tf.square(
                    1.0
                    - tf.to_float(self.global_step)
                    / tf.to_float(self.hps.warmup_steps)
                )
            ),
            lambda: tf.constant(1.0),
        )

        self.cons_multiplier = cons_multiplier * self.hps.max_cons_multiplier

        cons_mask = tf.equal(-1, self.labels)

        self.cons_loss = ssl_utils.diff_costs(
            self.hps.consistency_func,
            cons_mask,
            self.logits_student,
            self.logits_teacher,
            self.cons_multiplier,
        )

        self.ent_loss = ssl_utils.entropy_penalty(
            self.logits_student,
            tf.constant(self.hps.entropy_penalty_multiplier),
            cons_mask,
        )

        self.total_loss = self.labeled_loss + self.cons_loss + self.ent_loss

        with tf.control_dependencies(
            tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        ):
            self.train_op = tf.train.AdamOptimizer(
                learning_rate=self.lr
            ).minimize(self.total_loss, global_step=self.global_step)

        self.scalars_to_log = {
            "cons_multiplier": self.cons_multiplier,
            "accuracy": self.accuracy,
            "labeled_loss": self.labeled_loss,
            "cons_loss": self.cons_loss,
            "ent_loss": self.ent_loss,
            "total_loss": self.total_loss,
            "lr": self.lr,
        }

        for k, v in self.scalars_to_log.items():
            tf.summary.scalar(k, v)

    def prediction(self):
        """Actually get the outputs of the neural network."""

        network_function = functools.partial(
            self.network, is_training=self.is_training, hps=self.hps
        )
        inputs = self.processed_images
        with tf.variable_scope("prediction"):
            output = network_function(inputs, update_batch_stats=True)
        with tf.variable_scope("prediction", reuse=tf.AUTO_REUSE) as var_scope:
            if self.consistency_model == "mean_teacher":
                # Get list of all variables of the model, via prediction var scope
                model_vars = tf.get_collection(
                    "trainable_variables", "prediction"
                )
                # Apply EMA op to model variables and add it to graph updates
                ema = tf.train.ExponentialMovingAverage(self.hps.ema_factor)

                # Use custom getter which grabs EMA instead of raw variable
                def ema_getter(getter, name, *args, **kwargs):
                    var = getter(name, *args, **kwargs)
                    ema_var = ema.average(var)
                    return ema_var if ema_var else var

                var_scope.set_custom_getter(ema_getter)
                tf.add_to_collection(
                    tf.GraphKeys.UPDATE_OPS, ema.apply(model_vars)
                )

                # For mean teacher, we want to make the clean output look like
                # the EMA
                output_student = output
                output_teacher = network_function(inputs)
            elif self.consistency_model == "pi_model":
                # For pi-model, we want to make the output the same over two
                # passes
                output_student = output
                output_teacher = network_function(inputs)
            elif self.consistency_model == "vat":
                r_vadv = vat_utils.generate_virtual_adversarial_perturbation(
                    inputs, output, network_function, self.hps
                )

                # For VAT, we want to make the perturbed output similar to
                # clean
                output_student = network_function(inputs + r_vadv)
                output_teacher = output
            elif self.consistency_model == "pseudo_label":
                # Convert to probabilities so that we can use pseudo-label
                # threshold
                probs = tf.nn.softmax(output)
                # Get one-hot pseudo-label targets
                pseudo_labels = tf.one_hot(
                    tf.argmax(probs, axis=-1),
                    tf.shape(probs)[1],
                    dtype=probs.dtype,
                )
                # Masks denoting which data points have high-confidence predictions
                greater_than_thresh = tf.reduce_any(
                    tf.greater(probs, self.hps.pseudo_label_threshold),
                    axis=-1,
                    keepdims=True,
                )
                less_than_thresh = tf.logical_not(greater_than_thresh)

                output_student = output
                # Construct targets which are pseudo-labels when the prediction was
                # greater than threshold; vanilla logits otherwise. Multiplying the
                # one-hot pseudo_labels by 10 makes them look like logits.
                output_teacher = (
                    tf.cast(greater_than_thresh, probs.dtype)
                    * 10
                    * pseudo_labels
                    + tf.cast(less_than_thresh, probs.dtype) * output
                )
            else:
                assert False, "Unexpected consistency model {}".format(
                    self.consistency_model
                )

        # Don't let gradients flow through the teacher
        output_teacher = tf.stop_gradient(output_teacher)

        return output, output_student, output_teacher
