import matplotlib
#matplotlib.use("tkagg")
import matplotlib.pyplot as plt
import os, sys
import xarray as xr
import numpy as np
import tensorflow as tf
import pickle as p

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

class NDVI3G:
    def __init__(self, data_dir='./ndvi3g', patch_size=28,
                train_years=range(1981, 2010)):
        self.data_dir = data_dir
        allfiles = [os.path.join(self.data_dir, f) for f in os.listdir(self.data_dir) 
                        if f[-3:] == 'nc4']
        self.train_files = []
        self.valid_files = []
        self.scale = 1e-4
        for f in allfiles:
            year = int(f.split('/')[-1].split('.')[0].split('_')[-2])
            if year in train_years:
                self.train_files.append(f)
            else:
                self.valid_files.append(f)

        self.patch_size = patch_size

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

    def write_tfrecords(self, is_training): 
        dir = os.path.join(self.data_dir, 'tfrecords')
        if is_training:
            save_dir = os.path.join(dir, 'train')
        else:
            save_dir = os.path.join(dir, 'valid')

        writer = RecordWriter(save_dir)
        if is_training:
            files = self.train_files
        else:
            files = self.valid_files
        for f in files:
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

    def read_records(self, is_training=True, batch_size=1):
        dir = os.path.join(self.data_dir, 'tfrecords')
        with tf.name_scope("read_records"), tf.device("/cpu:0"):
            if is_training:
                buffer_size = 1000
                save_dir = os.path.join(dir, 'train')
            else:
                buffer_size = 1000
                save_dir = os.path.join(dir, 'valid')

            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            filenames = [os.path.join(save_dir, f) for f in os.listdir(save_dir)]
            if len(filenames) == 0:
                self.write_tfrecords(is_training)

            parser = lambda x: self._decode(x)
            dataset = tf.data.TFRecordDataset(filenames)
            dataset = dataset.map(parser)
            dataset = dataset.shuffle(buffer_size=buffer_size)
            dataset = dataset.batch(batch_size)
            dataset = dataset.make_one_shot_iterator()
            return dataset.get_next()


if __name__ == '__main__':
    ndvi = NDVI3G()
    # ndvi.plot_file(f=ndvi.train_files[0])
    for i, file in enumerate(ndvi.train_files):
        # print(os.path.join('./gimms',file[9:-4] + '.npz'))
        np.savez_compressed(os.path.join('./gimms',file[9:-4] + '.npz'), ndvi.make_subimages(ndvi.train_files[i]))
    #ndvi.plot_file() 
    el = ndvi.read_records(batch_size=10)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        d = sess.run(el)
        print(np.histogram(d.flatten()))

    print('Here')
