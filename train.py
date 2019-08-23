import os
import argparse
import pickle
from tqdm import tqdm
import PIL.Image
import numpy as np
import dnnlib.tflib as tflib
from encoder.generator_model import Generator
from encoder.nima import nima_reg
import tensorflow as tf
import random
from scipy.stats import truncnorm
import datetime
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image
import time

def split_to_batches(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]


def interpolate(dlatents, depth, bs):
    res = []
    t = list(dlatents)

    for i in range(bs):
        a = list(map(int, len(t) * np.random.uniform(0, 1, (2, 1))))
        a.sort()
        b = list(np.random.uniform(0, 1, (2, 1)))
        new_d = t[a[0]] * b[0] + t[a[1]] * b[1]
        k = depth - 1
        t.append(new_d)
        while k != 0:
            a = int(len(t) * random.uniform(0, 1))
            b = random.uniform(0, 1)
            new_d = t[-1] * b + t[a] * (1 - b)
            t.append(new_d)
            k -= 1
        res.append(t[-1])
    res = tf.reshape(tf.convert_to_tensor(np.vstack(res)), [2, 18, 512])
    return res


def combine(im_list, path):
    # images = map(PIL.Image.open, im_list)
    widths, heights = zip(*(i.size for i in im_list))

    total_width = sum(widths)
    max_height = max(heights)
    new_im = PIL.Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for im in im_list:
        new_im.paste(im, (x_offset, 0))
        x_offset += im.size[0]

    # new_im.save(path)
    new_im.save(path, 'PNG')


def interpolation(dlatent1, dlatent2):
    dlatents = [dlatent2]
    x = 0
    for i in range(4):
        x += 0.2
        dlatents.append(dlatent1 * x + (1 - x) * dlatent2)
    dlatents.append(dlatent1)
    return dlatents


def load_images(images_list, img_size):
    loaded_images = list()
    for img_path in images_list:
        img = image.load_img(img_path, target_size=(img_size, img_size))
        img = np.expand_dims(img, 0)
        loaded_images.append(img)
    loaded_images = np.vstack(loaded_images)
    preprocessed_images = preprocess_input(loaded_images)
    return preprocessed_images


def compute_loss(ref_img, ref_img_feature, interp, interp_features):
    interp_l1 = 0
    interp_loss = 0
    for k in range(2):  # n
        interp_l1 += 0.5 * tf.losses.mean_squared_error(ref_img[k], interp[0]) * 1e-5
        interp_loss += 0.5 * tf.losses.mean_squared_error(ref_img_feature[k], interp_features[0]) * 1e-6
    return interp_l1, interp_loss


def main():
    parser = argparse.ArgumentParser(description='Find latent representation of reference images using perceptual loss')
    # for now it's unclear if larger batch leads to better performance/quality
    parser.add_argument('--batch_size', default=2, help='Batch size for generator and perceptual model', type=int)
    parser.add_argument('--max_iteration', default=20000, help='max iterations for loss gd', type=int)
    parser.add_argument('--max_interpolation_depth', default=10, help='output tradeoff', type=int)

    # Perceptual model params
    parser.add_argument('--image_size', default=256, help='Size of images for perceptual model', type=int)
    parser.add_argument('--lr', default=2., help='Learning rate for perceptual model', type=float)
    parser.add_argument('--iterations', default=1000, help='Number of optimization steps for each batch', type=int)

    # Generator params
    parser.add_argument('--randomize_noise', default=False, help='Add noise to dlatents during optimization', type=bool)
    args, other_args = parser.parse_known_args()

    src_dir = '../stylegan_meta/data/anime'
    ref_images = [os.path.join(src_dir, x) for x in os.listdir(src_dir)]
    ref_images = list(filter(os.path.isfile, ref_images))

    # making dirs
    os.makedirs('reconst/', exist_ok=True)
    os.makedirs('dlatent/', exist_ok=True)
    os.makedirs('random/', exist_ok=True)
    os.makedirs('interpolation/', exist_ok=True)

    # Initialize generator and perceptual model
    sess = tflib.init_tf()
    with open('cache/263e666dc20e26dcbfa514733c1d1f81_karras2019stylegan-ffhq-1024x1024.pkl', "rb") as f:
        generator_network, discriminator_network, Gs_network = pickle.load(f)

    # generator
    generator = Generator(Gs_network, sess, args.batch_size, randomize_noise=args.randomize_noise)
    # VGG16
    vgg16 = VGG16(include_top=False, input_shape=(256, 256, 3))
    perceptual_model = Model(vgg16.input, vgg16.layers[9].output)

    # image placeholders
    generated_image = preprocess_input(tf.image.resize_images(generator.generated_image, (256, 256), method=1))
    generated_img_features = perceptual_model(generated_image)
    ref = tf.placeholder(dtype=tf.float32, shape=generated_image.shape)
    interp = tf.placeholder(dtype=tf.float32, shape=(1, 256, 256, 3))
    ref_f = tf.placeholder(dtype=tf.float32, shape=generated_img_features.shape)
    interp_f = tf.placeholder(dtype=tf.float32, shape=(1, 64,64,256))

    # losses
    ref_loss = tf.losses.mean_squared_error(np.ones(generated_img_features.shape) * ref_f,
                                             np.ones(generated_img_features.shape) * generated_img_features) / 92890.0
    ref_l1 = tf.losses.mean_squared_error(ref, generated_image) / 5890.0
    interp_l1, interp_loss = compute_loss(ref, ref_f, interp, interp_f)
    nima_loss = tf.math.abs(nima_reg(ref[0]) + nima_reg(ref[1])) / 2. - nima_reg(interp[0])) / 1.5
    loss = ref_loss + ref_l1 + interp_loss + interp_l1 + nima_loss
    loss2 = ref_loss + ref_l1 + nima_loss

    # variables
    variable = []
    tvars = tf.trainable_variables()
    for var in tvars:
        if "StyleMod" in var.name or 'learnable_dlatents' in var.name:
            variable.append(var)
    variable2 = []
    tvars = tf.trainable_variables()
    for var in tvars:
        if 'learnable_dlatents' in var.name:
            variable2.append(var)

    # optimizers
    optimizer = tf.train.GradientDescentOptimizer(1.5)
    min_op = optimizer.minimize(loss, var_list=variable)
    min_op2 = optimizer.minimize(loss2, var_list=variable2)

    # training loop
    epoch = 1
    itere = 1
    while True:
        for images_batch in tqdm(split_to_batches(ref_images, args.batch_size), total=len(ref_images) // args.batch_size):
            names = [os.path.splitext(os.path.basename(x))[0] for x in images_batch]
            oref = load_images(images_batch, 256)
            oref_f = perceptual_model.predict_on_batch(oref)
            for i in range(1000):
                ointerp = generator.get_interp()
                img = PIL.Image.fromarray(ointerp[0], 'RGB')
                img = img.resize((256, 256), PIL.Image.BILINEAR)
                ointerp = np.resize(preprocess_input(np.array(img)), (1, 256, 256, 3))
                ointerp_f = perceptual_model.predict_on_batch(ointerp)
                _ = sess.run([min_op],feed_dict={ref:oref, ref_f:oref_f, interp: ointerp, interp_f: ointerp_f})
                p_loss, l1, inter_loss, inter_l1 = sess.run([ref_loss, ref_l1, interp_loss, interp_l1],feed_dict={ref:oref, ref_f:oref_f, interp: ointerp, interp_f: ointerp_f})
                if i % 10 == 0:  
                    print(datetime.datetime.now())
                    print(i, p_loss, l1, inter_loss, inter_l1, sum([p_loss, l1, inter_loss, inter_l1]))

            # reconsruction
            dlatents = generator.get_dlatents()
            imgs = generator.generate_images()
            ointerp = generator.get_interp()
            for i, ii, name in zip(imgs, ointerp, names):
                img = PIL.Image.fromarray(i, 'RGB')
                img.save('reconst/{0}.png'.format(name), 'PNG')
                img = PIL.Image.fromarray(ii, 'RGB')
                img.save('reconst/{0}_interp.png'.format(name), 'PNG')
            print("reconst done", end="")
            print(datetime.datetime.now())

            # random
            d = interpolate(dlatents, 3, args.batch_size)
            imgs = generator.generate_images(d)
            w3 = truncnorm(-0.3, 0.3).rvs(args.batch_size * 18 * 512).astype("float32").reshape(args.batch_size, 18, 512)
            for i in imgs:
                img = PIL.Image.fromarray(i, 'RGB')
                img.save('random/{0}_inter.png'.format(itere), 'PNG')

            rand_img1 = generator.generate_images(w3)
            img1 = PIL.Image.fromarray(rand_img1[0], 'RGB')
            img1.save('random/{0}_w.png'.format(itere), 'PNG')
            print("random done", end="")
            print(datetime.datetime.now())

            # interpolationa
            if itere != 1:
                oref = load_images(ref_images[:args.batch_size], 256)
                oref_f = perceptual_model.predict_on_batch(oref)
                generator.reset_dlatents()
                for i in range(1000):
                    _, loss = generator.sess.run([min_op2, loss2], feed_dict={ref:oref, ref_f:oref_f})
            d1 = generator.get_dlatents()
            print("d1 done", end="")
            print(datetime.datetime.now())

            oref = load_images(ref_images[args.batch_size:2*args.batch_size], 256)
            oref_f = perceptual_model.predict_on_batch(oref)
            generator.reset_dlatents()
            for i in range(1000):
                _, loss = generator.sess.run([min_op2, loss2], feed_dict={ref: oref, ref_f: oref_f})
            d2 = generator.get_dlatents()
            print("d2 done", end="")
            print(datetime.datetime.now())
            d = interpolation(d1, d2)
            im_list = []
            for i in range(6):
                img = PIL.Image.fromarray(generator.generate_images(d[i])[0], 'RGB')
                im_list.append(img)
            combine(im_list, 'interpolation/{0}.png'.format(itere))

            itere += 1
            generator.reset_dlatents()
        if itere > args.max_iteration:
            break
        epoch += 1


if __name__ == "__main__":
    main()
