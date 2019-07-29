import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image
import keras.backend as K


def load_images(images_list, img_size):
    loaded_images = list()
    for img_path in images_list:
        img = image.load_img(img_path, target_size=(img_size, img_size))
        img = np.expand_dims(img, 0)
        loaded_images.append(img)
    loaded_images = np.vstack(loaded_images)
    preprocessed_images = preprocess_input(loaded_images)
    return preprocessed_images


def compute_loss(weight, mask, ref_img, ref_img_feature, interp, interp_features):
    interp_l1 = 0
    interp_loss = 0
    for i in range(ref_img.shape[0]):  # bs
        # n x 256 x 256 x 3
        ref_imgs = tf.convert_to_tensor([ref_img[k] for k in list(mask[i])], dtype=tf.float32)
        # n x 64 x 64 x 256
        ref_img_features = tf.convert_to_tensor([ref_img_feature[k] for k in list(mask[i])], dtype=tf.float32)

        for k in range(len(mask[i])):  # n
            interp_l1 += weight[i][k] * tf.losses.mean_squared_error(ref_imgs[k], interp[i]) * 1e-5
            interp_loss += weight[i][k] * tf.losses.mean_squared_error(ref_img_features[k], interp_features[i]) * 1e-6
    return interp_l1, interp_loss


class PerceptualModel:
    def __init__(self, img_size, layer=9, batch_size=1, sess=None):
        self.sess = tf.get_default_session() if sess is None else sess
        K.set_session(self.sess)
        self.img_size = img_size
        self.layer = layer
        self.batch_size = batch_size
        vgg16 = VGG16(include_top=False, input_shape=(self.img_size, self.img_size, 3))
        self.perceptual_model = Model(vgg16.input, vgg16.layers[self.layer].output)

        self.ref_img = None
        self.interp = None

        self.ref_img_features = None
        self.interp_features = None

        self.features_weight = None
        self.interp_mask = None
        self.weight = None

        self.loss = None
        self.L1 = None
        self.interp_loss = 0
        self.interp_L1 = 0

    def build_perceptual_model(self, generated_image_tensor):
        generated_image = preprocess_input(tf.image.resize_images(generated_image_tensor,
                                                                  (self.img_size, self.img_size), method=1))
        generated_img_features = self.perceptual_model(generated_image)

        # (bs, 64, 64, 256)
        self.ref_img_features = tf.get_variable('ref_img_features', shape=generated_img_features.shape,
                                                dtype='float32', initializer=tf.initializers.zeros())
        self.features_weight = tf.get_variable('features_weight', shape=generated_img_features.shape,
                                               dtype='float32', initializer=tf.initializers.zeros())
        # bs x 256 x 256 x 3
        self.ref_img = tf.get_variable('ref_img', shape=generated_image.shape,
                                       dtype='float32', initializer=tf.initializers.zeros())

        # bs x 256 x 256
        self.interp = tf.get_variable('interp', shape=generated_image.shape,
                                      dtype='float32', initializer=tf.initializers.zeros())
        self.interp_features = tf.get_variable('interp_features', shape=generated_img_features.shape,
                                               dtype='float32', initializer=tf.initializers.zeros())
        self.interp_mask = tf.get_variable('interp_mask', shape=(self.batch_size, 2),
                                           dtype='float32', initializer=tf.initializers.zeros())
        self.weight = tf.get_variable('weights', shape=(self.batch_size, 2),
                                      dtype='float32', initializer=tf.initializers.zeros())

        self.sess.run([self.ref_img_features.initializer, self.features_weight.initializer, self.ref_img.initializer,
                       self.interp.initializer, self.interp_mask.initializer,
                       self.interp_features.initializer, self.weight.initializer])

        self.loss = tf.losses.mean_squared_error(self.features_weight * self.ref_img_features,
                                                 self.features_weight * generated_img_features) / 92890.0
        self.L1 = tf.losses.mean_squared_error(self.ref_img, generated_image) / 5890.0

        # self.interp_L1, self.interp_loss = compute_loss(self.weight, self.interp_mask,
        # self.ref_img, self.ref_img_features,
        # self.interp, self.interp_features)

    def set_interpolation_images(self, images, mask, weights):
        assert (len(images) != 0 and len(images) <= self.batch_size)

        interpolated_image = preprocess_input(tf.image.resize_images(images, (self.img_size, self.img_size), method=1))
        interpolated_image = tf.saturate_cast(interpolated_image, tf.float32)
        interpolated_img_features = self.perceptual_model(interpolated_image)

        self.interp_L1, self.interp_loss = compute_loss(weights, mask,
                                                        self.ref_img, self.ref_img_features,
                                                        interpolated_image, interpolated_img_features)

        self.interp_mask.load(mask, self.sess)
        self.weight.load(weights, self.sess)
        self.interp.load(self.sess.run(interpolated_image), self.sess)
        self.interp_features.load(self.sess.run(interpolated_img_features), self.sess)

    def set_reference_images(self, images_list):
        assert (len(images_list) != 0 and len(images_list) <= self.batch_size)
        loaded_image = load_images(images_list, self.img_size)
        image_features = self.perceptual_model.predict_on_batch(loaded_image)
        weight_mask = np.ones(self.features_weight.shape)

        if len(images_list) != self.batch_size:
            features_space = list(self.features_weight.shape[1:])
            existing_features_shape = [len(images_list)] + features_space
            empty_features_shape = [self.batch_size - len(images_list)] + features_space

            existing_examples = np.ones(shape=existing_features_shape)
            empty_examples = np.zeros(shape=empty_features_shape)
            weight_mask = np.vstack([existing_examples, empty_examples])

            image_features = np.vstack([image_features, np.zeros(empty_features_shape)])

        self.features_weight.load(weight_mask, self.sess)
        self.ref_img_features.load(image_features, self.sess)
        self.ref_img.load(loaded_image, self.sess)

    def optimize(self, iterations, vars_to_optimize, learning_rate=1., interp=True):
        p_loss, l1, i_loss, i_l1 = 0, 0, 0, 0
        vars_to_optimize = vars_to_optimize if isinstance(vars_to_optimize, list) else [vars_to_optimize]
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        if interp:
            min_op = optimizer.minimize(self.loss + self.L1 + self.interp_loss + self.interp_L1,
                                        var_list=[vars_to_optimize])

            for i in range(iterations):
                _, p_loss, l1, i_loss, i_l1 = self.sess.run(
                    [min_op, self.loss, self.L1, self.interp_loss, self.interp_L1])
                #if i % 10 == 0:
                    #print(i, p_loss, l1, i_loss, i_l1, p_loss + l1 + i_loss + i_l1)
            return p_loss, l1, i_loss, i_l1

        else:
            min_op = optimizer.minimize(self.loss + self.L1, var_list=[vars_to_optimize])
            op_list = [min_op, self.loss, self.L1]

            for i in range(iterations):
                _, p_loss, l1 = self.sess.run(op_list)
                #if i % 10 == 0:
                    #print(i, p_loss, l1, p_loss + l1)
            return p_loss, l1

