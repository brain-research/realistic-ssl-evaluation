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

"""Converts pickled imagenet_32x32 files to .npy files.

The default imagenet_32x32 data files are stored in Python 3
pickly encoding.

The rest of our code is in Python 2, so we have a separate script that
just deals with this issue separately.

You can execute it as:

python3 convert_imagenet.py

after which you should no longer have to manually deal with imagenet.
"""
import os
import numpy as np
import pickle


_DATA_DIR = "data/imagenet_32/"


def unpickle(filename):
    with open(filename, "rb") as fo:
        dict = pickle.load(fo)
    return dict


train_file_names = ["train_data_batch_" + str(idx) for idx in range(1, 11)]
val_file_names = ["val_data"]
for file_name in train_file_names + val_file_names:
    data = unpickle(os.path.join(_DATA_DIR, file_name))
    image_file_name = file_name + "_image"
    label_file_name = file_name + "_label"
    np.save(os.path.join(_DATA_DIR, image_file_name), data["data"])
    np.save(os.path.join(_DATA_DIR, label_file_name), np.array(data["labels"]))
