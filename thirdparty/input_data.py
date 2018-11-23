# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# This file has been modified from the original code released by Google Inc.
# in the Tensorflow library. The Apache License for this file
# can be found in the directory thirdparty.
# The modifications in this file are subject to the same Apache License, Version 2.0.
#
# Copyright 2018 D-Wave Systems Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Functions for downloading and reading Binarized MNIST data."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import numpy as np
import os
import tensorflow as tf
import scipy.io
import xarray as xr
import matplotlib.pyplot as plt

import numpy

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

class RecordWriter:
    def __init__(self, dir):
        self.dir = dir
        self.filecounter = 0

        if not os.path.exists(self.dir):
            os.makedirs(self.dir)

        self.filename = os.path.join(self.dir, 'part-%3i.tfrecords')

    def write_records(self, data):
        writer = tf.python_io.TFRecordWriter(self.filename % self.filecounter)
        n, height, width = data.shape
        for d in data:
            features = {'ndvi': _bytes_feature(d.astype(np.float32).tostring())}
            example = tf.train.Example(features=tf.train.Features(feature=features))
            writer.write(example.SerializeToString())
        writer.close()
        self.filecounter += 1

class Datasets():
    def __init__(self, data_dir, batch_size=32, eval_batch_size=32, nTestYears=10, patch_size=28, scale = 1e-4, binarize_threshold=0.4):
        test_years = range(1981, 1981+nTestYears)
        train_years = range(1981+nTestYears, 1981+nTestYears+10)
        validation_years = range(1981+nTestYears+10, 2010)

        ### Data selection etc
        self.binarize_threshold = binarize_threshold
        self.data_dir = data_dir
        allfiles = [os.path.join(self.data_dir, f) for f in os.listdir(self.data_dir)
                    if f[-3:] == 'nc4']
        self.train_files = []
        self.validation_files = []
        self.test_files = []

        for f in allfiles:
            year = int(f.split('/')[-1].split('.')[0].split('_')[-2])
            if year in train_years:
                self.train_files.append(f)
            elif year in validation_years:
                self.validation_files.append(f)
            elif year in test_years:
                self.test_files.append(f)

        self.train = dataset(train_years, self.train_files, data_dir, id='train', patch_size=patch_size, scale=scale, batch_size=batch_size)
        self.test = dataset(test_years, self.test_files, data_dir, id='test', patch_size=patch_size, scale=scale, batch_size=eval_batch_size)
        self.validation = dataset(validation_years, self.validation_files, data_dir, id='valid', patch_size=patch_size, scale=scale, batch_size=eval_batch_size)

class dataset():
    def __init__(self, years, files, data_dir, id=None, patch_size=28, scale=1e-4, should_binarize=True, batch_size=32):
        self.batch_size = batch_size

        self.should_binarize = should_binarize
        self.data_dir = data_dir
        self.id = id
        if id == 'test':
            self.id = 'valid'
        self.years = years
        self.patch_size = patch_size

        if id == 'train':
            self._num_examples = 2024218
            ### These are the factors of the number of examples
            # 1, 2, 7, 14, 191, 382, 757, 1337, 1514, 2674, 5299, 10598, 144587, 289174, 1012109, 2024218
        else:
            self._num_examples = 426169
            ### These are the factors of the number of examples
            # 1, 229, 1861, 426169
        #self._num_examples = self.get_num_examples()
        self.scale = scale
        self.files = files

        self._epochs_completed = 0
        self._index_in_epoch = 0

        self.create_tfrecords()

    def create_tfrecords(self):
        self.read_records()

    def num_examples(self):
        # self._num_examples = images.shape[0]
        return

    @property
    def images(self):
        return self._images

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, sess):
        """Return the next `batch_size` examples from this data set."""
        ### Only used to get mean x
        data = sess.run(self.read_records())
        return data

    def get_num_examples(self):
        dir = os.path.join(self.data_dir, 'tfrecords', self.id)
        tf_records_filenames = os.listdir(dir)
        c = 0
        for fn in tf_records_filenames:
            tmp = os.path.join(dir, fn)
            for record in tf.python_io.tf_record_iterator(tmp):
                c += 1
            # print('File %s' % fn)

        print('num_examples = %i' % c)
        return c

    def _open_file(self, f):
        fill_value = -32768.
        ds = xr.open_dataset(f)
        ds = ds.where(ds.ndvi != fill_value)
        ds['ndvi'] = ds['ndvi'] * self.scale
        #ds['ndvi'].values[ds['ndvi'].values == fill_value] = np.nan
        return ds

    def plot_file(self, f=None):
        if f is None:
            f = self.files[0]
        ds = self._open_file(f)
        fig, axs = plt.subplots(3,4)
        axs = np.ravel(axs)
        for t in range(12):
            axs[t].imshow(ds.ndvi.isel(time=t).values)
            axs[t].axis('off')
        plt.show()

    def make_subimages(self, f):
        size = self.patch_size
        ds = self._open_file(f)
        data = []
        for i, t in enumerate(ds.time.values):
            vals = ds['ndvi'].sel(time=t).values
            height, width = vals.shape
            hsteps = int(height / size)
            wsteps = int(width / size)
            for h in range(hsteps):
                for w in range(wsteps):
                    v = vals[h*size:size*(h+1), w*size:size*(w+1)]
                    if np.all(np.isfinite(v)):
                        data.append(v)
        return np.array(data)

    def write_tfrecords(self):
        dir = os.path.join(self.data_dir, 'tfrecords')
        save_dir = os.path.join(dir, self.id)
        writer = RecordWriter(save_dir)
        for f in self.files:
            print(f)
            data = self.make_subimages(f)
            writer.write_records(data)

    def _decode(self, serialized_example):
        feature_dict = {'ndvi': tf.FixedLenFeature([], tf.string)}
        features = tf.parse_single_example(serialized_example, features=feature_dict)
        ndvi = tf.decode_raw(features['ndvi'], tf.float32)
        ndvi = tf.reshape(ndvi, [self.patch_size, self.patch_size, 1])
        ndvi = tf.cast(ndvi, tf.float32, name='ndvi')
        return ndvi

    def read_records(self):
        dir = os.path.join(self.data_dir, 'tfrecords')
        with tf.name_scope("read_records"), tf.device("/cpu:0"):
            buffer_size = 1000
            save_dir = os.path.join(dir, self.id)

            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            filenames = [os.path.join(save_dir, f) for f in os.listdir(save_dir)]
            if len(filenames) == 0:
                self.write_tfrecords()

            parser = lambda x: self._decode(x)
            dataset = tf.data.TFRecordDataset(filenames)
            dataset = dataset.map(parser)
            dataset = dataset.shuffle(buffer_size=buffer_size)
            dataset = dataset.batch(self.batch_size)
            dataset = dataset.make_one_shot_iterator()
            self._images = dataset.get_next()
            return self._images

    def flatten_flatten(self, x):
        data = []
        im = []
        for image in x:
            for column in image:
                for row in column:
                    pixel = row[0]
                    im.append(pixel)
            data.append(im)
        return (np.asarray(data, dtype=np.float32), None)
