import os
import argparse
import pickle
from tqdm import tqdm
import PIL.Image
import numpy as np
import dnnlib.tflib as tflib
from encoder.generator_model import Generator
from encoder.perceptual_model import PerceptualModel
import tensorflow as tf
import random
from scipy.stats import truncnorm
import datetime

def split_to_batches(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]


def make_mask(bs, n):
    """
    make a bs x n index matrix
    """
    mask = []
    weight = []
    for i in range(bs):
        a = list(map(int, bs * np.random.uniform(0, 1, (n, 1))))
        a.sort()
        b = list(map(float, bs * np.random.uniform(0, 1, (n, 1))))
        b_sum = sum(b)
        b = [x / b_sum for x in b]
        mask.append(a)
        weight.append(b)
    mask = np.vstack(mask)
    weight = np.vstack(weight)
    return mask, weight


def interpolate(dlatents, depth, bs):
    """
    interpolate the dlatents depth many times
    """
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
    res = tf.convert_to_tensor(np.vstack(res))
    return res


def combine(im_list, path):
    """combine images """
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
    """interpolate between two images (6)"""
    dlatents = [dlatent2]
    x = 0
    for i in range(4):
        x += 0.2
        dlatents.append(dlatent1 * x + (1 - x) * dlatent2)
    dlatents.append(dlatent1)
    return dlatents


def main():
    parser = argparse.ArgumentParser(description='Find latent representation of reference images using perceptual loss')
    # parser.add_argument('src_dir', help='Directory with images for encoding')

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

    ref_images = [os.path.join('./data/anime', x) for x in os.listdir('./data/anime')]
    ref_images = list(filter(os.path.isfile, ref_images))

    print("number of example: {0}".format(len(ref_images)))

    os.makedirs('reconst/', exist_ok=True)
    os.makedirs('dlatent/', exist_ok=True)
    os.makedirs('random/', exist_ok=True)
    os.makedirs('interpolation/', exist_ok=True)

    # get pretrained model
    tflib.init_tf()
    with open('cache/263e666dc20e26dcbfa514733c1d1f81_karras2019stylegan-ffhq-1024x1024.pkl', "rb") as f:
        generator_network, discriminator_network, Gs_network = pickle.load(f)

    generator = Generator(Gs_network, args.batch_size, randomize_noise=args.randomize_noise)
    perceptual_model = PerceptualModel(args.image_size, layer=9, batch_size=args.batch_size)
    perceptual_model.build_perceptual_model(generator.generated_image)

    epoch = 1
    itere = 1
    while True:
        for images_batch in tqdm(split_to_batches(ref_images, args.batch_size),
                                 total=len(ref_images) // args.batch_size):
            names = [os.path.splitext(os.path.basename(x))[0] for x in images_batch]
            interp_mask, mask_weight = make_mask(args.batch_size, 2)
            perceptual_model.set_reference_images(images_batch)

            # vars to be trained
            variable = []
            tvars = tf.trainable_variables()
            for var in tvars:
                if "StyleMod" in var.name or 'learnable_dlatents' in var.name:
                    variable.append(var)

            for i in range(1000):
                # get interpolated dlatents
                interpolated_imgs = generator.get_interp(interp_mask, mask_weight)
                print("now set".format(datetime.datetime.now()))
                #TODO set interpolated image, this becomes slower as training goes
                perceptual_model.set_interpolation_images(interpolated_imgs, interp_mask, mask_weight)
                print("now opt".format(datetime.datetime.now()))
                #TODO this is slow too, but was fine before adding interpolation loss
                p_loss, l1, inter_loss, inter_l1 = perceptual_model.optimize(1, variable, learning_rate=args.lr,
                                                                             interp=True)
                print("now gen interp".format(datetime.datetime.now()))
                # ten iteration I examined the loss
                # Adding interpolation loss makes the original loss harder to opt/original loss goes down very slowly
                if i % 10 == 0:
                    print(datetime.datetime.now())
                    imgs = generator.generate_images()
                    img = PIL.Image.fromarray(imgs[0], 'RGB')
                    img.save('reconst/k_{0}.png'.format(i), 'PNG')
                    print("epoch: {0}, iterations: {1}, loss: {2}, {3}, {4}, {5}, {6}".format(epoch, itere, p_loss, l1,
                                                                                              inter_loss, inter_l1,
                                                                                              p_loss + l1 + inter_loss + inter_l1))
            # reconsruction
            dlatents = generator.get_dlatents()
            imgs = generator.generate_images()
            for i, name in zip(imgs, names):
                img = PIL.Image.fromarray(i, 'RGB')
                img.save('reconst/{0}.png'.format(name), 'PNG')
            print("reconst done", end="")
            print(datetime.datetime.now())

            # random
            d = interpolate(dlatents, 3, args.batch_size)
            imgs = generator.generate_images(d)
            w3 = truncnorm(-0.3, 0.3).rvs(2 * 18 * 512).astype("float32").reshape(2, 18, 512)
            for i in imgs:
                img = PIL.Image.fromarray(i, 'RGB')
                img.save('random/{0}_inter.png'.format(itere), 'PNG')

            rand_img1 = generator.generate_images(w3)
            img1 = PIL.Image.fromarray(rand_img1[0], 'RGB')
            img1.save('random/{0}_w.png'.format(itere), 'PNG')
            print("random done", end="")
            print(datetime.datetime.now())

            # interpolation
            if itere != 1:
                perceptual_model.set_reference_images(ref_images[:args.batch_size])
                generator.reset_dlatents()
                perceptual_model.optimize(1000, generator.dlatent_variable, learning_rate=args.lr, interp=False)
            d1 = generator.get_dlatents()
            print("d1 done", end="")
            print(datetime.datetime.now())
            perceptual_model.set_reference_images(ref_images[args.batch_size:2 * args.batch_size])
            generator.reset_dlatents()
            perceptual_model.optimize(1000, generator.dlatent_variable, learning_rate=args.lr, interp=False)
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
