import os
import numpy as np
import tensorflow as tf
from PIL import Image
import encoder.vgg19 as vgg
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.layers import Dense, Dropout
from keras.models import Model

def get_nima_model(input=None):
    base_model = InceptionResNetV2(input_shape=(None, None, 3), include_top=False, pooling='avg', weights=None,
                                   input_tensor=input)
                                   x = Dropout(0.75)(base_model.output)
                                   x = Dense(10, activation='softmax')(x)
                                   
                                   model = Model(base_model.input, x)
                                   model.load_weights('encoder/inception_resnet_weights.h5')
                                   return model


def preprocess(image):
    return (image / 127.5) - 1.0


def postprocess(image):
    return (image + 1.0) * 127.5


def compute_nima_loss(image):
    model = get_nima_model(image)
    
    def mean_score(scores):
        scores = tf.squeeze(scores)
        si = tf.range(1, 11, dtype=tf.float32)
        return tf.reduce_sum(tf.multiply(si, scores), name='nima_score')
    
    nima_score = mean_score(model.output)
    
    nima_loss = tf.identity(10.0 - nima_score, name='nima_loss')
    return nima_loss

def nima_reg(tensor):
    tensor = tf.reshape(tensor, [1, tensor.shape[0], tensor.shape[1],tensor.shape[2]])
    transfer_image_nima = preprocess(tensor)
    nima_loss = compute_nima_loss(transfer_image_nima)
    return nima_loss
