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

"""Builds label maps for semisupervised learning.

In more detail:

We are performing semi-supervised learning for image classification.
That means that some of the training inputs to our image classifier will
have labels, and some of them won't.

In order to approximate the instance where we have lots of unlabeled data and
a small amount of labeled data, we take large labeled datasets and throw
out some of the labels.

However, it would take lots of disk space and impede our ability to make
changes to the process if we tried to create all of these dataset by saving new
subsets of the existing image classification datasets.

Instead, we throw out labels on the fly.
We accomplish this by creating these label maps, which are lists of unique
example indices which we will treat as corresponding to labeled images.
If an image is not in the given label map, we treat it as being unlabeled.

At training time, these indices are read from disk and used to construct
tensorflow hash-tables, so that we can strip out the labels quickly for the
necessary images. Saving these images on disk also makes it easier to reproduce
our experiments.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import random

from tensorflow.python.platform import app
from tensorflow.python.platform import flags
from tensorflow.python.platform import gfile
from absl import logging

from lib import dataset_utils
from lib import paths


flags.DEFINE_string("dataset_name", "default", "Name of source dataset.")
flags.DEFINE_integer(
    "n_labeled_min", 10, "Least number of labeled examples to use."
)
flags.DEFINE_integer(
    "n_labeled_max", 100, "Most number of labeled examples to use (inclusive)."
)
flags.DEFINE_integer("n_labeled_step", 10, "Increment to change n_labeled by.")
flags.DEFINE_integer(
    "label_map_copies", 2, "Number of random maps to generate per label count."
)
flags.DEFINE_string(
    "fkeys_path",
    paths.LABEL_MAP_PATH,
    "Where to write read the fkeys and write the label_maps.",
)
flags.DEFINE_string(
    "imagenet_path",
    paths.RAW_IMAGENET_PATH,
    "Where to read raw imagenet files.",
)

FLAGS = flags.FLAGS


def main(_):
    # Build a label map for each label_count, with several seeds
    for n_labeled in range(
        FLAGS.n_labeled_min,
        FLAGS.n_labeled_max + FLAGS.n_labeled_step,
        FLAGS.n_labeled_step,
    ):
        for label_map_index in range(FLAGS.label_map_copies):
            build_single_label_map(
                n_labeled,
                label_map_index,
                FLAGS.dataset_name,
                FLAGS.imagenet_path,
                FLAGS.fkeys_path,
            )


def build_single_label_map(
    n_labeled, label_map_index, dataset_name, imagenet_path, fkeys_path
):
    """Builds just one label map - we call this in a larger loop.

    As a side effect, this function writes the label map to a file.

    Args:
        n_labeled: An integer representing the total number of labeled
            examples desired.
        label_map_index: An integer representing the index of the label map.
            We may want many label_maps w/ same value of n_labeled.
            This allows us to disambiguate.
        dataset_name: A string representing the name of the dataset.
            One of 'cifar10', 'svhn', 'imagenet'.
        imagenet_path: A string that encodes the location of the raw imagenet
            data.
        fkeys_path: A string that encodes where to read fkeys from and write
            label_maps to.

    Raises:
        ValueError: if passed an unrecognized dataset_name.
    """
    # Set the name of the label_map
    # This will be named as label_map_count_{count}_idx_{idx}
    destination_name = "label_map_count_{}_index_{}".format(
        n_labeled, label_map_index
    )
    result_dict = {"values": []}
    n_labeled_per_class = (
        n_labeled // dataset_utils.DATASET_CLASS_COUNT[dataset_name]
    )

    if dataset_name == "imagenet":
        synsets = gfile.ListDir(imagenet_path)
        for synset in synsets:
            logging.info("processing: %s", synset)
            unique_ids = [
                f[: -len(".JPEG")]
                for f in gfile.ListDir(os.path.join(imagenet_path, synset))
            ]
            result_dict["values"] += random.sample(
                unique_ids, n_labeled_per_class
            )
    elif dataset_name in {"cifar10", "svhn"}:
        path = os.path.join(fkeys_path, dataset_name, "label_to_fkeys_train")
        with gfile.GFile(path, "r") as f:
            label_to_fkeys = json.load(f)
        for label in label_to_fkeys.keys():
            result_dict["values"] += random.sample(
                label_to_fkeys[label], n_labeled_per_class
            )
    else:
        raise ValueError("Dataset not supported: {}.".format(dataset_name))

    # Save the results in a JSON file
    result_path = os.path.join(fkeys_path, dataset_name, destination_name)
    with gfile.GFile(result_path, "w") as f:
        json.dump(result_dict, f)


if __name__ == "__main__":
    app.run(main)
