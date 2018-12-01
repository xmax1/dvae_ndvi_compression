# Copyright 2018 D-Wave Systems Inc.
# DVAE# licensed to authorized users only under the applicable license
# agreement.  See LICENSE.

import os

import numpy as np
import tensorflow as tf

from thirdparty import input_data
from train_eval import run_training
from vae import VAE

flags = tf.app.flags
flags.DEFINE_string('data_dir', '/home/max/quantumml/repos/dvae/tj/compression/ndvi3g',
                    'Directory to save training/test data.')
flags.DEFINE_string('log_dir', './logs',
                    'Directory to save the checkpoints and summaries.')
flags.DEFINE_string('dataset', 'gimms',
                    'Dataset to run experiments. We support "omniglot" and "binarized_mnist" for now. AND GIMMS')
flags.DEFINE_string('struct', '1-layer-nonlin',
                    'Structure used in the encoder. Select one from: 1-layer-lin, 1-layer-nonlin, '
                    '2-layer-nonlin, 4-layer-nonlin')
flags.DEFINE_string('baseline', 'dvaes_power',
                    'Baseline used in training. Select one from: '
                    'dvae_spike_exp for DVAE (spike and exp), '
                    'dvaepp_exp for DVAE++ (exponential), '
                    'dvaepp_power for DVAE++ (power), '
                    'dvaes_gi for DVAE# (Gaussian Integral), '
                    'dvaes_gauss for DVAE# (Gaussian), '
                    'dvaes_exp for DVAE# (exponential), '
                    'dvaes_unexp for DVAE# (uniform+exponential), '
                    'dvaes_power for DVAE# (power).')

flags.DEFINE_string('sampler_name', 'qupa',
                     'qupa, pcd or dwave are the optional strings Whether or not use Quadrant\'s population annealing (PA) for sampling. '
                     'Setting this flag to False will enable PCD sampling.')

flags.DEFINE_integer('k', 1,
                     'Number of samples used for estimating the variational bound in the case of DVAE/DVAE++'
                     'and the importance weighted bound in the case of DVAE#.')
flags.DEFINE_float('beta', None,
                   'This parameter is set automatically. Setting this flag will overwrite the default values.')
flags.DEFINE_integer('num_train_iter', int(1e4),
                     'Number of training iterations (steps).')
# For the factors of the dataset size go to input_data.Datasets. The batchsizes must be a factor
flags.DEFINE_integer('train_batchsize', 382,
                     'Training batch size')
flags.DEFINE_integer('eval_batchsize', 229,
                     'Evaluation batch size')
flags.DEFINE_integer('eval_iw_samples', 4000,
                     'Number of importance weighted samples used in the final evaluation')
flags.DEFINE_float('base_lr', 3e-3, 'Base learning rate')
flags.DEFINE_integer('input_size', 784, 'Number of pixels in the vectorized input.')

flags.DEFINE_integer('num_latent_units', 64,
                     'Number of latent variables')

FLAGS = flags.FLAGS

def get_config(mean_x):
    dataset = FLAGS.dataset
    base_lr = FLAGS.base_lr
    num_iter = FLAGS.num_train_iter
    eval_iw_samples = FLAGS.eval_iw_samples
    batch_size = FLAGS.train_batchsize
    k = FLAGS.k
    baseline = FLAGS.baseline
    struct = FLAGS.struct
    sampler_name = FLAGS.sampler_name
    num_latent_units = FLAGS.num_latent_units

    # set config train
    config_train = {'dataset': dataset, 'mean_x': mean_x, 'lr': base_lr, 'num_iter': num_iter, 'k_iw': eval_iw_samples,
                    'batch_size': batch_size, 'k': k, 'data_dir': FLAGS.data_dir, 'sampler_name': sampler_name, 'num_latent_units':num_latent_units}

    # The following sets beta
    if FLAGS.beta is not None:
        beta = FLAGS.beta
    elif dataset == 'binarized_mnist':
        config_train['eval_batch_size'] = 1000 if FLAGS.eval_batchsize is None else FLAGS.eval_batchsize
        if struct == '1-layer-lin':
            beta = {'dvae_spike_exp': 5., 'dvaepp_exp': 12., 'dvaepp_power': 40.,
                    'dvaes_gi': 40., 'dvaes_gauss': 40.,
                    'dvaes_exp': 12., 'dvaes_unexp': 40., 'dvaes_power': 40.}[baseline]
        elif struct == '1-layer-nonlin':
            beta = {'dvae_spike_exp': 5., 'dvaepp_exp': 14., 'dvaepp_power': 30.,
                    'dvaes_gi': 30., 'dvaes_gauss': 30.,
                    'dvaes_exp': 12., 'dvaes_unexp': 30., 'dvaes_power': 40.}[baseline]
        elif struct in {'2-layer-nonlin', '4-layer-nonlin'}:
            beta = {'dvae_spike_exp': 4., 'dvaepp_exp': 10., 'dvaepp_power': 30.,
                    'dvaes_exp': 10., 'dvaes_unexp': 16.,
                    'dvaes_gauss': 25., 'dvaes_gi': 30., 'dvaes_power': 30.}[baseline]
        else:
            raise NotImplementedError('The struct %s is not supported' % struct)

    elif dataset == 'omniglot':
        config_train['eval_batch_size'] = 269 if FLAGS.eval_batchsize is None else FLAGS.eval_batchsize
        if struct == '1-layer-lin':
            beta = {'dvae_spike_exp': 4., 'dvaepp_exp': 14., 'dvaepp_power': 40.,
                    'dvaes_gi': 40., 'dvaes_gauss': 40.,
                    'dvaes_exp': 12., 'dvaes_unexp': 40., 'dvaes_power': 40.}[baseline]
        elif struct == '1-layer-nonlin':
            beta = {'dvae_spike_exp': 4., 'dvaepp_exp': 12., 'dvaepp_power': 40.,
                    'dvaes_gi': 30., 'dvaes_gauss': 25.,
                    'dvaes_exp': 10., 'dvaes_unexp': 16., 'dvaes_power': 30.}[baseline]
        elif struct in {'2-layer-nonlin', '4-layer-nonlin'}:
            beta = {'dvae_spike_exp': 4., 'dvaepp_exp': 10., 'dvaepp_power': 30.,
                    'dvaes_gi': 25., 'dvaes_gauss': 25.,
                    'dvaes_exp': 8., 'dvaes_unexp': 16., 'dvaes_power': 20.}[baseline]
        else:
            raise NotImplementedError('The struct %s is not supported' % struct)

    elif dataset == 'gimms':
        config_train['eval_batch_size'] = 229 if FLAGS.eval_batchsize is None else FLAGS.eval_batchsize
        if struct == '1-layer-lin':
            beta = {'dvae_spike_exp': 5., 'dvaepp_exp': 12., 'dvaepp_power': 40.,
                    'dvaes_gi': 40., 'dvaes_gauss': 40.,
                    'dvaes_exp': 12., 'dvaes_unexp': 40., 'dvaes_power': 40.}[baseline]
        elif struct == '1-layer-nonlin':
            beta = {'dvae_spike_exp': 5., 'dvaepp_exp': 14., 'dvaepp_power': 30.,
                    'dvaes_gi': 30., 'dvaes_gauss': 30.,
                    'dvaes_exp': 12., 'dvaes_unexp': 30., 'dvaes_power': 40.}[baseline]
        elif struct in {'2-layer-nonlin', '4-layer-nonlin'}:
            beta = {'dvae_spike_exp': 4., 'dvaepp_exp': 10., 'dvaepp_power': 30.,
                    'dvaes_exp': 10., 'dvaes_unexp': 16.,
                    'dvaes_gauss': 25., 'dvaes_gi': 30., 'dvaes_power': 30.}[baseline]
        else:
            raise NotImplementedError('The struct %s is not supported' % struct)
    else:
        raise NotImplementedError('The dataset %s is not supported' % dataset)

    # set encoder settings based on struct
    if struct == '1-layer-lin':
        num_latent_layers = 1
        num_det_layers = 0
        num_latent_units = FLAGS.num_latent_units
    elif struct == '1-layer-nonlin':
        num_latent_layers = 1
        num_det_layers = 2
        num_latent_units = FLAGS.num_latent_units
    elif struct == '2-layer-nonlin':
        num_latent_layers = 2
        num_det_layers = 2
        num_latent_units = FLAGS.num_latent_units
    elif struct == '4-layer-nonlin':
        num_latent_layers = 4
        num_det_layers = 2
        num_latent_units = FLAGS.num_latent_units
    else:
        raise NotImplementedError('The struct %s is not supported' % struct)

    config = {'dist_type': baseline, 'weight_decay': 1e-4, 'num_latent_layers': num_latent_layers,
              'num_latent_units': num_latent_units, 'name': 'lay0',
              'num_det_layers_enc': num_det_layers, 'num_det_units_enc': 200, 'weight_decay_enc': 1e-4,
              'beta': beta, 'sampler_name':sampler_name, 'batch_norm': True}

    config_recon = {'num_det_layers': num_det_layers, 'num_det_units': 200, 'weight_decay_dec': 1e-4,
                    'name': 'recon', 'batch_norm': True}

    # set importance weight training option
    if baseline in {'dvae_spike_exp', 'dvaepp_exp', 'dvaepp_power'}:
        config_train['use_iw'] = False
    elif config_train['k'] == 1:
        config_train['use_iw'] = False
    else:
        config_train['use_iw'] = True

    expr_id = '%s/%s/%s/%d' % (dataset, struct, baseline, k)
    config_train['expr_id'] = expr_id

    return config_train, config, config_recon


def main(argv):
    datasets = input_data.Datasets(FLAGS.data_dir, batch_size=FLAGS.train_batchsize, eval_batch_size=FLAGS.eval_batchsize)

    # We need per pixel mean for normalizing input. This is from the original code. It is unclear if this is working as rqd
    # with tf.Session() as sess:
    #     im = np.reshape(datasets.train.next_batch(sess), (-1, 28, 28))
    # sess.close()
    mean_x = None

    # get configurations
    config_train, config, config_recon = get_config(mean_x)
    log_dir = os.path.join(FLAGS.log_dir, config_train['expr_id'])

    vae = VAE(num_input=FLAGS.input_size, config=config, config_recon=config_recon, config_train=config_train)
    run_training(vae, cont_train=False, config_train=config_train, log_dir=log_dir)


if __name__ == '__main__':
    tf.app.run()

