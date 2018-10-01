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

"""A set of small TF utility functions shared between train and evaluate."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def hash_float(x, big_num=1000 * 1000):
    """Hash a tensor 'x' into a floating point number in the range [0, 1)."""
    return tf.cast(
        tf.string_to_hash_bucket_fast(x, big_num), tf.float32
    ) / tf.constant(float(big_num))


def make_set_filter_fn(elements):
    """Constructs a TensorFlow "set" data structure.

    Note that sets returned by this function are uninitialized. Initialize them
    by calling `sess.run(tf.tables_initializer())`

    Args:
        elements: A list of non-Tensor elements.

    Returns:
        A function that when called with a single tensor argument, returns
        a boolean tensor if the argument is in the set.
    """
    table = tf.contrib.lookup.HashTable(
        tf.contrib.lookup.KeyValueTensorInitializer(
            elements, tf.tile([1], [len(elements)])
        ),
        default_value=0,
    )

    return lambda x: tf.equal(table.lookup(x), 1)


def filter_fn_from_comma_delimited(string):
    """Parses a string of comma delimited numbers, returning a filter function.

    This utility function is useful for parsing flags that represent a set of
    options, where the default is "all options".

    Args:
        string: e.g. "1,2,3", or empty string for "set of all elements"

    Returns:
        A function that when called with a single tensor argument, returns
        a boolean tensor that evaluates to True if the argument is in the set.
        If 'string' argument is None, the set is understood to contain all
        elements and the function always returns a True tensor.
    """
    if string:
        return make_set_filter_fn(list(map(int, string.split(","))))
    else:
        return lambda x: tf.constant(True)
