
# Copyright 2018 D-Wave Systems Inc.
# QuPA licensed to authorized users only under the applicable license
# agreement.  See LICENSE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle as p
import math
import tensorflow as tf
import numpy as np
import time
from thirdparty.qRBM.qRBM import qRBM_


class quantum_rbm(object):

    def __init__(self, left_size, right_size, num_samples,
                 name=None, dtype=None):
        """Constructor.

        PCD instances hold a Tensorflow variable containing RBM
        samples.

        Args:
            left_size: Number of left-side variable in the RBM.
            right_size: Number of right-side variable in the RBM.
            num_samples: Number of samples.
            name: Tensorflow variable scope name.
            dtype: Data type of samples and parameters, either
                tf.float32 or tf.float64.
        """

        if dtype is None:
            dtype = tf.float32

        self.left_size = left_size
        self.right_size = right_size
        sample_size = left_size + right_size
        # default name is iterated if called again and is PCD otherwise
        with tf.variable_scope(name, default_name='PCD'):
            self._samples = tf.get_variable('samples', trainable=False,
                                            initializer=tf.round(
                                                tf.random_uniform(
                                                    [num_samples, sample_size],
                                                    dtype=dtype)))
            self._logz = tf.get_variable('log_z', shape=[], trainable=False,
                                         initializer=tf.constant_initializer(
                                             sample_size * math.log(2),
                                             dtype=dtype))

        self.nodes = range(sample_size)
        self.quantum_sampler = qRBM_(self.nodes, gauge=None, batch_size=64, lr = 0.0001, load=False, binarize_threshold=None,
                 method_sampler='C16',
                 h_init=[], J_init=[], inv_BETA=0.1 )

    @property
    def samples_var(self):
        """Tensorflow variable containing samples."""
        return self._samples

    @property
    def log_z_var(self):
        """Tensorflow variable containing current log Z estimate.

        This variable is always returned by PCD.training_log_z() (with
        gradients attached).  Note that nothing in this class ever
        updates this variable and you are free to update it with log Z
        estimates obtained elsewhere.  It is provided for compatibility
        with qupa.PopulationAnnealer.
        """
        return self._logz

    def samples(self):
        """Returns current samples."""
        return self._samples.read_value()

    def sampler_pyfunc(self, samples, biases, weights,):
        with tf.name_scope("quantum_samples"):
            # print('samples', samples, 'biases', biases, 'weights', weights, 'sweeps', num_mcmc_sweeps)
            new_samples = tf.py_func(self.anneal, [samples, biases, weights], tf.float32)
            return samples.assign(new_samples)

    def anneal(self, samples, biases, weights):

        nreads = len(samples)
        self.quantum_sampler.h_graph = biases
        J_graph = np.zeros((len(self.nodes), len(self.nodes)))
        J_graph[:self.left_size, self.right_size:] = weights
        self.quantum_sampler.J_graph = J_graph
        new_samples = self.quantum_sampler.sample(nreads=nreads, answer_mode='raw')
        new_samples = new_samples.astype(np.float32)

        return new_samples


    def training_log_z(self, biases, weights, num_mcmc_sweeps=1, name=None):
        """Updates samples and provides log Z gradient.

        This function always returns the value of self.log_z_var.
        While this value isn't interesting and has nothing to do with
        the partition function of the RBM, it does provide gradients
        with respect to biases and weights.


        Args:
            biases: A tensor of shape [L+R] containing RBM biases (left
                side then right side).
            weights: A tensor of shape [L, R] containing RBM weights.
            num_mcmc_sweeps: Number of MCMC sweeps to run.
            name: Name for the Tensorflow name_scope created.

        Data types of biases and weights must match dtype provided to
        PCD.__init__().

        Returns:
            A scalar tensor whose value is self.log_z_var.
        """

        with tf.name_scope(name, default_name='training_log_z'):
            new_samples = self.sampler_pyfunc(self._samples, biases, weights)
            with tf.get_default_graph().gradient_override_map(
                    {'IdentityN': '_TrainingLogZGrad'}):
                # indentity_n is within the mcmc function somehow
                # Initialised the pcd and the only output was the identity_n
                return tf.identity_n([self._logz, biases, weights, new_samples])[0]

    def save(self, path, x):
        with open(path, 'wb') as f:
            p.dump(x, f)

