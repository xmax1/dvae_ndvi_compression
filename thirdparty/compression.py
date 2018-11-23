import tensorflow as tf
import numpy as np
import os
from dist_util import FactorialBernoulliUtil

def compress(sess, data_dir, input_data, epoch, vae, num_latent_units):
    compression_dir = os.path.join(data_dir, 'compression', '%i_latent_units' % num_latent_units)

    if not os.path.exists(compression_dir):
        os.makedirs(compression_dir)
    is_training = False

    ### Input
    data_input = sess.run(input_data)
    np.save(os.path.join(compression_dir, 'epoch%i_input.npy' % epoch), data_input)

    ### Compression
    _, post_samples_compressed = vae.encoder.hierarchical_posterior(input_data, is_training)
    # convert list of samples to single tensor
    post_samples_concat = tf.concat(axis=-1, values=post_samples_compressed)
    compressed = sess.run(post_samples_concat)
    np.save(os.path.join(compression_dir, 'epoch%i_compressed.npy' % epoch ), compressed)

    ### Decompression
    # create features for the likelihood p(x|z)
    x_output_activations = vae.decoder.reconstruct(post_samples_concat, is_training)
    # add data bias
    x_output_activations[0] = x_output_activations[0] + vae.train_bias
    # form the output dist util.
    x_output_dist = FactorialBernoulliUtil(x_output_activations)
    # create the final output
    x_output = tf.nn.sigmoid(x_output_dist.logit_mu)
    output = sess.run(x_output)
    np.save(os.path.join(compression_dir, 'epoch%i_output.npy' % epoch ), output)

    l2_norm = np.sqrt((data_input - output)**2)
    np.save(os.path.join(compression_dir, 'epoch%i_l2_norm.npy' % epoch ), l2_norm)
    return
