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

"""Generic utils for all datasets."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict
import json
import os
import random as rand
from absl import flags
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
from lib import paths

# Constants used for dealing with the files, matches convert_to_records.
FLAGS = flags.FLAGS
TRAIN_FILE = "train.tfrecords"
VALIDATION_FILE = "validation.tfrecords"
TEST_FILE = "test.tfrecords"

FILES = {"train": TRAIN_FILE, "valid": VALIDATION_FILE, "test": TEST_FILE}

DATASET_SHAPE = {
    "cifar10": (None, 32, 32, 3),
    "cifar_unnormalized": (None, 32, 32, 3),
    "svhn": (None, 32, 32, 3),
    "svhn_extra": (None, 32, 32, 3),
    "imagenet_32": (None, 32, 32, 3),
}
DATASET_DTYPE = {
    "cifar10": tf.float32,
    "cifar_unnormalized": tf.uint8,
    "svhn": tf.uint8,
    "svhn_extra": tf.uint8,
    "imagenet_32": tf.uint8,
}
DATASET_CLASS_COUNT = {
    "cifar10": 10,
    "cifar_unnormalized": 10,
    "svhn": 10,
    "svhn_extra": 10,
    "imagenet_32": 1000,
}
DATASET_EXAMPLE_COUNT = {
    "train": {
        "cifar10": 50000 - 5000,
        "cifar_unnormalized": 50000 - 5000,
        "svhn": 73257 - 7326,
        "svhn_extra": 531131,
        "imagenet_32": 1281167 - 50050,
    },
    "test": {
        "cifar10": 10000,
        "cifar_unnormalized": 10000,
        "svhn": 26032,
        "imagenet_32": 50000,
    },
    "valid": {
        "cifar10": 5000,
        "cifar_unnormalized": 5000,
        "svhn": 7326,
        "imagenet_32": 50050,
    },
}


def int64_feature(value):
    """Create a feature that is serialized as an int64."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def bytes_feature(value):
    """Create a feature that is stored on disk as a byte array."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to(images, labels, num_examples, name, directory, dataset_name):
    """Converts a dataset to tfrecords.

    This function has the side effect of writing a dataset to disk.

    Args:
        images (tensor): many images.
        labels (tensor): many labels.
        num_examples (int): how many images and labels we are converting.
        name (str): the base name we will give to the file we construct.
        directory (str): where the dataset will be written.
        dataset_name (str): the name of the actual dataset, e.g. 'svhn'.

    Raises:
        ValueError: if the image size and label size don't match.
    """

    if images.shape[0] != num_examples:
        raise ValueError(
            "Images size %d does not match label size %d."
            % (images.shape[0], num_examples)
        )
    rows, cols, depth = images.shape[1:4]

    label_to_fkeys = defaultdict(list)

    filename = os.path.join(directory, dataset_name, name + ".tfrecords")
    if not os.path.exists(os.path.join(directory, dataset_name)):
        os.makedirs(os.path.join(directory, dataset_name))
    tf.logging.info("Writing {}".format(filename))
    writer = tf.python_io.TFRecordWriter(filename)
    for index in range(num_examples):
        image_raw = images[index].tostring()
        file_key = str(index)
        file_key_bytes = file_key.encode()
        label = int(labels[index])
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    "height": int64_feature(rows),
                    "width": int64_feature(cols),
                    "depth": int64_feature(depth),
                    "label": int64_feature(label),
                    "file_key": bytes_feature(file_key_bytes),
                    "image_raw": bytes_feature(image_raw),
                }
            )
        )
        writer.write(example.SerializeToString())

        label_to_fkeys[label].append(file_key)

    writer.close()

    # Serialize map from fkey to label
    # This way, we can create balanced semisupervised datasets later
    result_path = os.path.join(
        directory, dataset_name, "label_to_fkeys_" + name
    )
    with tf.gfile.GFile(result_path, "w") as f:
        json.dump(label_to_fkeys, f)


def build_simple_mixed_batch_datasets(
    labeled_dataset_name, unlabeled_dataset_name, labeled_parser
):
    """Build the datasets for parsed labeled and unlabeled data.

    Args:
        labeled_dataset_name (str): name of the labeled dataset.
        unlabeled_dataset_name (str): name of the unlabeled dataset.
        labeled_parser (func): Function that parsers one image from labeled
            dataset.

    Returns:
        A pair of datasets (labeled_dataset, unlabeled_dataset)

    Raises:
        ValueError: If the datasets aren't compatible
    """
    # If split were not train, this function would not be called.
    split = "train"

    labeled_filenames = get_filenames(labeled_dataset_name, split)
    unlabeled_filenames = get_filenames(unlabeled_dataset_name, split)

    unlabeled_parser = construct_parser(unlabeled_dataset_name)

    # A dataset object holding only the labeled examples from primary.
    labeled_dataset = tf.data.TFRecordDataset(labeled_filenames)
    labeled_dataset = labeled_dataset.map(
        labeled_parser, num_parallel_calls=32
    ).prefetch(100)

    # A dataset object holding all examples from secondary, unlabeled.
    unlabeled_dataset = tf.data.TFRecordDataset(unlabeled_filenames)
    unlabeled_dataset = unlabeled_dataset.map(
        unlabeled_parser, num_parallel_calls=32
    ).prefetch(100)

    if are_datasets_compatible(labeled_dataset_name, unlabeled_dataset_name):
        return [labeled_dataset, unlabeled_dataset]
    else:
        raise ValueError(
            "Datasets {}, {} not compatible",
            labeled_dataset_name,
            unlabeled_dataset_name,
        )


def parse_small_example(
    dataset, serialized_example, image_shape, apply_normalization
):
    """Parses an example from one of the smaller datasets.

  This function also performs some pre-processing.

  Args:
    dataset (str): the name of the dataset.
    serialized_example (tf.Example): blob representing a tf.Example.
    image_shape (int): the size we want the image to have.
    apply_normalization (int): whether to mean-normalize the images.

  Returns:
    A tuple (image, label, file_key), where:
    * image is a single image.
    * label is an int32 with the true label,
    * fkey is a key that will be used to decide whether to perturb the label
      for the purpose of performing semi-supervised learning.
  """
    features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
            "image_raw": tf.FixedLenFeature([], tf.string),
            "label": tf.FixedLenFeature([], tf.int64),
            "file_key": tf.FixedLenFeature([], tf.string),
        },
    )

    # Read the bytes that constitute the image
    image_dtype = DATASET_DTYPE[dataset]
    image = tf.decode_raw(features["image_raw"], image_dtype)
    image.set_shape(np.prod(image_shape))

    if apply_normalization:
        # Convert from [0, 255] -> [-1., 1.] floats.
        image = tf.cast(image, tf.float32)
        image = image * (1. / 255) - 0.5
        image *= 2.

    # Reshape the images
    image = tf.reshape(image, image_shape)

    # Convert label from a scalar uint8 tensor to an int32 scalar.
    label = tf.cast(features["label"], tf.int32)

    # Read the file_key, which we use for label manipulation
    file_key = features["file_key"]

    return image, label, file_key


def construct_label_table(dataset, label_map):
    """Given a label_map file, construct the hash-table it represents.

  Args:
    dataset (str): the name of the dataset.
    label_map (str): the label_map filename.

  Returns:
    A tensorflow hashtable from String to Bool.
  """

    # We just want to use all the labels in this case.
    if label_map is None:
        return None
    # So that we don't need to store copies of label_maps for all our different
    # imagenet variations:
    if "imagenet" in dataset:
        dataset = "imagenet"

    # Load the map from disk
    result_path = os.path.join(paths.TRAIN_DIR, dataset, label_map)
    with tf.gfile.GFile(result_path, "r") as f:
        result_dict = json.load(f)

    # Because the imagenet keys are just the image filenames:
    if "imagenet" in dataset:
        keys = [st + ".JPEG" for st in result_dict["values"]]
    else:
        # This result_dict will contain a list of file_keys
        # The json library seems to default to loading them as unicode?
        # so I ascii encode them (I wrote them, so I know they only include
        # ascii characters)
        keys = [key.encode("ascii", "ignore") for key in result_dict["values"]]
    values = [True] * len(keys)
    label_table = tf.contrib.lookup.HashTable(
        tf.contrib.lookup.KeyValueTensorInitializer(keys, values), False
    )
    return label_table


def get_filenames(dataset_name, split):
    """Get the names of the tfrecord files for this (dataset, split) pair.

  Args:
    dataset_name (str): Name of the dataset, e.g. svhn.
    split (str): Which split to use, e.g. test.

  Returns:
    A list of filenames.
  Raises:
      ValueError: if the dataset or split is not supported.
  """
    if dataset_name in [
        "cifar10",
        "svhn",
        "cifar_unnormalized",
        "imagenet_32",
    ]:
        filenames = [os.path.join(paths.TRAIN_DIR, dataset_name, FILES[split])]
    elif dataset_name == "svhn_extra":
        if split != "train":
            raise ValueError("svhn_extra dataset only has a train split")
        filenames = [os.path.join(paths.TRAIN_DIR, "svhn", "extra.tfrecords")]
    else:
        raise ValueError("Unsupported dataset, split pair.")
    return filenames


def are_datasets_compatible(labeled_dataset_name, unlabeled_dataset_name):
    """Check if a pair of datasets are compatible for semisupevised learning.

  Args:
    labeled_dataset_name (str): a string identifier.
    unlabeled_dataset_name (str): a string identifier.
  Returns:
    Boolean
  """
    valid_combos = [
        ("cifar_unnormalized", "svhn"),
        ("svhn", "cifar_unnormalized"),
        ("svhn", "svhn_extra"),
    ]
    return (labeled_dataset_name == unlabeled_dataset_name) or (
        labeled_dataset_name,
        unlabeled_dataset_name,
    ) in valid_combos


def shuffle_merge(dataset_1, dataset_2):
    """Merge two tensorflow dataset objects in a more shuffley way.

  If the two datasets being merged repeat indefinitely,
  an iterator created on this dataset will alternate examples
  from the two datasets.

  If the datasets have very different sizes, this might not
  be the behavior that you want.

  If you want examples to come out in proportion to the original
  size of the dataset, you will need to do something like store
  some marker of the source dataset and then do rejection resampling
  on the outputs to rebalance them according to source dataset size.

  Args:
    dataset_1 (Tensorflow dataset): first dataset object.
    dataset_2 (Tensorflow dataset): second dataset object.
  Returns:
    A new dataset.
  """
    zipped = tf.data.Dataset.zip((dataset_1, dataset_2))

    def concat_func(x_1, x_2):
        return tf.data.Dataset.from_tensors(x_1).concatenate(
            tf.data.Dataset.from_tensors(x_2)
        )

    alternated = zipped.flat_map(concat_func)
    return alternated


def construct_parser(dataset_name):
    """Construct a parser based on configuration data.

  Args:
    dataset_name (str): Name of the dataset, e.g. 'svhn'.

  Returns:
    A parser of serialized tf.Examples.
  """
    image_shape = DATASET_SHAPE[dataset_name][1:]
    image_size = image_shape[1]
    apply_normalization = (
        dataset_name != "cifar10"
    )  # cifar10 is pre-ZCA normalized

    def parser(example):
        return parse_small_example(
            dataset_name, example, image_shape, apply_normalization
        )

    return parser


def get_dataset(dataset_name, split):
    """Returns a tf.data.Dataset for a supported dataset and split."""
    filenames = get_filenames(dataset_name, split)
    parser = construct_parser(dataset_name)
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(parser, num_parallel_calls=32).prefetch(100)
    return dataset


def gcn(images, multiplier=55, eps=1e-10):
    """Performs global contrast normalization on a numpy array of images.

    Args:
        images: Numpy array representing the original images.
            This function expects a rank-2 array (no H, C, W).
        multiplier: Post-normalization multiplier.
        eps: Small number for numerical stability.

    Returns:
        A numpy array of the same shape as images, but normalized.
    """
    images = images.astype(float)
    # Subtract the mean of image
    images -= images.mean(axis=1, keepdims=True)
    # Divide out the norm of each image
    per_image_norm = np.linalg.norm(images, axis=1, keepdims=True)
    # Avoid divide-by-zero
    per_image_norm[per_image_norm < eps] = 1
    return multiplier * images / per_image_norm


def tf_gcn(inp, multiplier=55., eps=1e-8):
    """Performs global contrast normalization on a TF tensor of images.

    Args:
        inp: Numpy array representing the original images.
            This function expects a rank-4 array [N, H, W, C].
        multiplier: Float governing severity of the adjustment.
        eps: Float governing numerical stability.
    Returns:
        Tensor with same shape as inp.
    """
    inp -= tf.reduce_mean(inp, axis=[1, 2, 3], keepdims=True)
    denominator = tf.sqrt(
        tf.reduce_sum(tf.square(inp), axis=[1, 2, 3], keepdims=True)
    )
    denominator /= multiplier
    denominator = tf.where(
        tf.less(denominator, tf.constant(eps)),
        tf.ones_like(denominator),
        denominator,
    )
    inp /= denominator
    return inp


def get_zca_transformer(images, identity_scale=0.1, eps=1e-10, root_path=None):
    """Creates function performing ZCA normalization on a numpy array.

    Args:
        images: Numpy array of flattened images, shape=(n_images, n_features)
        identity_scale: Scalar multiplier for identity in SVD
        eps: Small constant to avoid divide-by-zeor
        root_path: Optional path to save the ZCA params to.

    Returns:
        A function which applies ZCA to an array of flattened images
    """
    image_covariance = np.cov(images, rowvar=False)
    U, S, _ = np.linalg.svd(
        image_covariance + identity_scale * np.eye(*image_covariance.shape)
    )
    zca_decomp = np.dot(U, np.dot(np.diag(1. / np.sqrt(S + eps)), U.T))
    image_mean = images.mean(axis=0)

    if root_path is not None:
        mean_path = os.path.join(root_path, "zca_mean")
        decomp_path = os.path.join(root_path, "zca_decomp")
        if not os.path.exists(root_path):
            os.makedirs(root_path)
        np.save(mean_path, image_mean)
        np.save(decomp_path, zca_decomp)

    return lambda x: np.dot(x - image_mean, zca_decomp)


def zca_normalize(image_batch, array_path):
    """ZCA normalizes a batch of images stored in a TF tensor.

  Args:
    image_batch: TF Tensor of shape [N, H, W, C].
    array_path: Location of saved numpy arrays.

  Returns:
    A normalized batch of images.
  """

    mean_path = os.path.join(array_path, "zca_mean.npy")
    with gfile.GFile(mean_path, "rb") as f:
        averages = np.load(f)

    decomp_path = os.path.join(array_path, "zca_decomp.npy")
    with gfile.GFile(decomp_path, "rb") as f:
        decomposition = np.load(f)

    _, height, width, channels = image_batch.shape
    # Transpose back into the shape for which ZCA statistics were computed
    # in build_tfrecords.py
    image_batch = tf.transpose(image_batch, [0, 3, 1, 2])
    image_batch = tf.reshape(image_batch, [-1, height * width * channels])
    averages_tensor = tf.constant(averages, dtype=tf.float32)
    decomposition_tensor = tf.constant(decomposition, dtype=tf.float32)
    results = tf.matmul(image_batch - averages_tensor, decomposition_tensor)
    results = tf.reshape(results, [-1, channels, height, width])
    results = tf.transpose(results, [0, 2, 3, 1])
    return results
