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

"""The one-stop API for getting batches of data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from lib import dataset_utils

_PREFETCH_BUFFER_SIZE = 1000


def get_simple_mixed_batch(
    labeled_dataset_name,
    unlabeled_dataset_name,
    split,
    batch_size,
    shuffle_buffer_size,
    labeled_data_filter_fn=None,
    unlabeled_data_filter_fn=None,
    mode="mix",
):
    """A less flexible, more memory-efficient version of get_simple_mixed_batch.

    Always mixes data from a fully labeled dataset with a fully unlabeled
    dataset.  Each batch will have half labeled, half unlabeled items.

    Args:
        labeled_dataset_name (str): Name of the dataset from which all labeled
            data comes.
        unlabeled_dataset_name (str): Name of the dataset from which all
            unlabeled data comes.
        split (str): Which split to use, e.g. test/train/valid.
        batch_size (int): Number of examples per returned batch.
        shuffle_buffer_size (int): How big the shuffle buffers are.
        labeled_data_filter_fn (function): Function to decide which labeled
            data to look at. Takes three tensor arguments: image, label, file
            key.  Returns a boolean tensor. Defaults to no filter (all images).
        unlabeled_data_filter_fn (function): Function to decide which unlabeled
            data to look at. Same signature as labeled_data_filter.
        mode (str): "labeled" - use only labeled data,
                    "mix" (default) -  use mixed data

    Returns:
        A tuple (images, labels, batch_count, remainder, num_classes), where:
            * images is a tensor of images
            * labels is an int32 tensor with shape [batch_size] with the true
                label, a number in the range [-1, num_classes).
            * batch_count is an int representing the number of batches in an
                epoch.
            * remainder is an int representing how many elements will be left
                over at the end of the epoch. We use this for eval to make sure
                that we've evaluated all the elements in the dataset exactly
                once.
            * num_classes is an int - it gives the number of classes in the
                dataset.

    Raises:
        ValueError: if the arguments are incompatible.
    """

    if labeled_data_filter_fn is None:
        labeled_data_filter_fn = lambda image, label, fkey: True
    if unlabeled_data_filter_fn is None:
        unlabeled_data_filter_fn = lambda image, label, fkey: True

    image_shape = dataset_utils.DATASET_SHAPE[labeled_dataset_name][1:]
    example_count = dataset_utils.DATASET_EXAMPLE_COUNT[split][
        labeled_dataset_name
    ]
    batch_count = example_count // batch_size
    remainder = example_count % (batch_size * batch_count)
    num_classes = dataset_utils.DATASET_CLASS_COUNT[labeled_dataset_name]
    labeled_parser = dataset_utils.construct_parser(labeled_dataset_name)

    tf.logging.info("Using multi-dataset feed.")

    # Here we construct the three constituent datasets that will make up
    # our final mixed dataset:
    [
        labeled_dataset,
        unlabeled_dataset,
    ] = dataset_utils.build_simple_mixed_batch_datasets(
        labeled_dataset_name=labeled_dataset_name,
        unlabeled_dataset_name=unlabeled_dataset_name,
        labeled_parser=labeled_parser,
    )

    # Filter before shuffling, so that our shuffle buffer only contains
    # examples that we want to actually see in the output dataset.
    # Otherwise we're not shuffling as much, by wasting effective shuffle
    # buffer size on images that get shuffled but then thrown away.
    labeled_dataset = labeled_dataset.filter(labeled_data_filter_fn)
    unlabeled_dataset = unlabeled_dataset.filter(unlabeled_data_filter_fn)

    # Forget labels on unlabeled dataset by setting "label" to -1. This is
    # necessary in order to use labeled datasets as unlabeled data, such as
    # SVHN's "extra" split.
    unlabeled_dataset = unlabeled_dataset.map(
        lambda image, label, fkey: (image, tf.constant(-1), fkey)
    )

    # We need to repeat all the datasets before shuffle_merging them.
    # Shuffle_merge involves a zip, which will truncate the shorter of two
    # finite datasets. We also just want them repeated at the end of the
    # pipeline anyway.
    labeled_dataset = labeled_dataset.cache().repeat()
    labeled_dataset = labeled_dataset.shuffle(shuffle_buffer_size, 0)
    unlabeled_dataset = unlabeled_dataset.cache().repeat()
    unlabeled_dataset = unlabeled_dataset.shuffle(shuffle_buffer_size, 0)

    # These operations merge the datasets in a way that intersperses
    # elements from each, rather than just concatenating them.
    if labeled_dataset_name == "imagenet_32" or mode == "labeled":
        # Don't waste batch space for imagenet pre-training.
        dataset = labeled_dataset
    elif mode == "unlabeled":
        dataset = unlabeled_dataset
    else:
        assert mode == "mix"
        dataset = dataset_utils.shuffle_merge(
            labeled_dataset, unlabeled_dataset
        )

    # Batch the results
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(_PREFETCH_BUFFER_SIZE)
    # Get the actual results from the iterator
    # images, labels, fkeys = dataset.make_one_shot_iterator().get_next()
    iterator = dataset.make_initializable_iterator()
    tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS, iterator.initializer)
    images, labels, fkeys = iterator.get_next()

    images = tf.reshape(images, (batch_size,) + image_shape)
    labels = tf.reshape(labels, [batch_size])

    # Summarize the images and labels as they come out of the provider
    tf.summary.image("images_in_provider", images)
    tf.summary.histogram("labels_in_provider", labels)

    return images, labels, fkeys, batch_count, remainder, num_classes
