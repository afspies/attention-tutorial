# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Tetrominoes dataset reader."""

import tensorflow as tf
from functools import partial

COMPRESSION_TYPE = tf.io.TFRecordOptions.get_compression_type_string('GZIP')
IMAGE_SIZE = [35, 35]
# The maximum number of foreground and background entities in the provided
# dataset. This corresponds to the number of segmentation masks returned per
# scene.
MAX_NUM_ENTITIES = 4
BYTE_FEATURES = ['mask', 'image']

# Create a dictionary mapping feature names to `tf.Example`-compatible
# shape and data type descriptors.
features = {
    'image': tf.io.FixedLenFeature(IMAGE_SIZE+[3], tf.string),
    'mask': tf.io.FixedLenFeature([MAX_NUM_ENTITIES]+IMAGE_SIZE+[1], tf.string),
    'x': tf.io.FixedLenFeature([MAX_NUM_ENTITIES], tf.float32),
    'y': tf.io.FixedLenFeature([MAX_NUM_ENTITIES], tf.float32),
    'shape': tf.io.FixedLenFeature([MAX_NUM_ENTITIES], tf.float32),
    'color': tf.io.FixedLenFeature([MAX_NUM_ENTITIES, 3], tf.float32),
    'visibility': tf.io.FixedLenFeature([MAX_NUM_ENTITIES], tf.float32),
}


def _decode(example_proto, get_masks=True):
  # Parse the input `tf.Example` proto using the feature description dict above.
  single_example = tf.io.parse_single_example(example_proto, features)

  tensors_to_decode = BYTE_FEATURES if get_masks else ['image']
  for k in BYTE_FEATURES:
    single_example[k] = tf.squeeze(tf.io.decode_raw(single_example[k], tf.uint8),
                                   axis=-1)

  return single_example if get_masks else single_example['image']


def dataset(tfrecords_path, read_buffer_size=None, map_parallel_calls=None, get_masks=True):
  """Read, decompress, and parse the TFRecords file.

  Args:
    tfrecords_path: str. Path to the dataset file.
    read_buffer_size: int. Number of bytes in the read buffer. See documentation
      for `tf.data.TFRecordDataset.__init__`.
    map_parallel_calls: int. Number of elements decoded asynchronously in
      parallel. See documentation for `tf.data.Dataset.map`.

  Returns:
    An unbatched `tf.data.TFRecordDataset`.
  """
  raw_dataset = tf.data.TFRecordDataset(
      tfrecords_path, compression_type=COMPRESSION_TYPE,
      buffer_size=read_buffer_size)
  partial_decode_fn = partial(_decode, get_masks=get_masks)
  raw_dataset = raw_dataset.map(partial_decode_fn, num_parallel_calls=map_parallel_calls).cache()
  return raw_dataset


