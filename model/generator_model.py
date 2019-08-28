import tensorflow as tf
import numpy as np
import dnnlib.tflib as tflib
from functools import partial


def create_stub(name, batch_size):
    return tf.constant(0, dtype='float32', shape=(batch_size, 0))


def create_variable_for_generator(name, batch_size):
    return tf.get_variable('learnable_dlatents',
                           shape=(batch_size, 18, 512),
                           dtype='float32',
                           initializer=tf.initializers.random_normal())


class Generator:
    def __init__(self, model, sess, batch_size, randomize_noise=False):
        self.batch_size = batch_size
        self.model = model
        self.mapping = self.model.components.mapping
        self.synthesis = self.model.components.synthesis

        self.initial_dlatents = np.zeros((self.batch_size, 18, 512))
        self.synthesis.run(self.initial_dlatents,
                           randomize_noise=randomize_noise, minibatch_size=self.batch_size,
                           custom_inputs=[partial(create_variable_for_generator, batch_siz$
                                          partial(create_stub, batch_size=batch_size)],
                           structure='fixed')

        self.sess = sess
        self.graph = tf.get_default_graph()

        self.dlatent_variable = next(v for v in tf.global_variables() if 'learnable_dlatents' in v.name)
        self.set_dlatents(self.initial_dlatents)

        self.generator_output = self.graph.get_tensor_by_name('G_synthesis_1/_Run/concat:0')
        self.generated_image = tflib.convert_images_to_uint8(self.generator_output, nchw_to_nhwc=True, uint8_cast=False)
        self.generated_image_uint8 = tf.saturate_cast(self.generated_image, tf.uint8)

    def reset_dlatents(self):
        self.set_dlatents(self.initial_dlatents)

    def set_dlatents(self, dlatents):
        assert (dlatents.shape == (self.batch_size, 18, 512))
        self.dlatent_variable.load(dlatents, self.sess)

    def get_dlatents(self):
        return self.sess.run(self.dlatent_variable)

    def generate_images(self, dlatents=None):
        if dlatents is not None:
            self.set_dlatents(dlatents)
        return self.sess.run(self.generated_image_uint8)
                                          
    def get_interp(self):
        d = self.get_dlatents()
        # n, 18, 512
        interp_d = np.tile(np.mean(d, axis=0).reshape((1, 18, 512)), (2, 1, 1)).reshape((2, 18, 512))
        imgs = self.generate_images(interp_d)
        self.set_dlatents(d)
        return imgs

    """
    def get_interp(self, interp_mask, mask_weight):
        """get the interpolated images"""
        d = self.get_dlatents()
        # n, 18, 512
        interp_d = []
        for i in range(self.batch_size):
            d_source = d[list(interp_mask[i])]
            k = sum([x * w for x, w in zip(d_source, mask_weight[i])])
            interp_d.append([k])
        interp_d = np.vstack(interp_d)
        imgs = self.generate_images(interp_d)
        self.set_dlatents(d)
        return imgs
