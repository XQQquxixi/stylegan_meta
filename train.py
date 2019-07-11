import os
import argparse
import tensorflow as tf
from loss.Loss import LossModel
import pickle
import dnnlib
import dnnlib.tflib as tflib
import config
import PIL.Image
import numpy as np
from models.FGAN import Generator
from tqdm import tqdm


def split_to_batches(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]


def main():
    parser = argparse.ArgumentParser(description='Find latent representation of reference images using perceptual loss')
    #parser.add_argument('src_dir', default='./data', help='Directory with images for encoding')
    #parser.add_argument('generated_images_dir', default='./gen', help='Directory for storing generated images')
    #parser.add_argument('dlatent_dir', default='./dgen', help='Directory for storing dlatent representations')

    # for now it's unclear if larger batch leads to better performance/quality
    parser.add_argument('--batch_size', default=1, help='Batch size for generator and perceptual model', type=int)

    # Perceptual model params
    parser.add_argument('--image_size', default=256, help='Size of images for perceptual model', type=int)
    parser.add_argument('--lr', default=1., help='Learning rate for perceptual model', type=float)
    parser.add_argument('--iterations', default=1000, help='Number of optimization steps for each batch', type=int)

    # Generator params
    parser.add_argument('--randomize_noise', default=False, help='Add noise to dlatents during optimization', type=bool)
    args, other_args = parser.parse_known_args()

# ----------------------------------------------------------------------
    ref_images = [os.path.join('./data/anime', x) for x in os.listdir('./data/anime')]
    ref_images = list(filter(os.path.isfile, ref_images))

    if len(ref_images) == 0:
        raise Exception('%s is empty' % './data/anime')

    os.makedirs('./gen', exist_ok=True)
    os.makedirs('./dgen', exist_ok=True)

# ----------------------------------------------------------------------
    # Initialize generator and perceptual model
    tflib.init_tf()
    with dnnlib.util.open_url('https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ', cache_dir=config.cache_dir) as f:
        generator_network, discriminator_network, Gs_network = pickle.load(f)

    generator = Generator(Gs_network, args.batch_size, randomize_noise=args.randomize_noise)
    loss_model = LossModel(args.image_size, layer=9, batch_size=args.batch_size)
    loss_model.build_loss_model(generator.generated_image)

    # Optimize (only) dlatents by minimizing perceptual loss between reference and generated images in feature space
    for images_batch in tqdm(split_to_batches(ref_images, args.batch_size), total=len(ref_images) // args.batch_size):
        names = [os.path.splitext(os.path.basename(x))[0] for x in images_batch]
        loss_model.set_reference_images(images_batch)
        variable = generator.dlatent_variable + tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="StyleMod")
        op = loss_model.optimize(variable, iterations=args.iterations, learning_rate=args.lr)
        # Generate images from found dlatents and save them
        generated_images = generator.generate_images()
        generated_dlatents = generator.get_dlatents()
        for img_array, dlatent, img_name in zip(generated_images, generated_dlatents, names):
            img = PIL.Image.fromarray(img_array, 'RGB')
            img.save(os.path.join('./gen', f'{img_name}.png'), 'PNG')
            np.save(os.path.join('./dgen', f'{img_name}.npy'), dlatent)

        generator.reset_dlatents()


if __name__ == "__main__":
    main()





"""
    # start trainig loop
    losses = AverageMeter()
    print_freq = args.print_freq
    eval_freq = args.eval_freq
    save_freq = eval_freq
    max_iteration = args.iters

    iteration = 0
    epoch = 0
    while True:
        # Iterate over dataset (one epoch).
        for data in dataloader:
            img = data[0]
            indices = data[1]
            latents = model.embeddings(indices)
            eps = tf.random_uniform(latents.size()) * 0.001
            # forward
            img_generated = model.forward(latents + eps)
            loss = criterion.forward(img_generated, img, latents, model.trainable)
            losses.update(loss.item(), img.size(0))

            # compute gradient and do SGD step
            gen_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="StyleMod")

            if iteration % print_freq == 0:
                temp = "train loss: %0.5f " % loss.item()
                temp += "| avg loss %0.5f " % losses.avg
                print(iteration, temp)
                losses = AverageMeter()

            if iteration % eval_freq == 0 and iteration > 0:
                img_prefix = os.path.join(checkpoint_dir, "%d_" % iteration)
                generate_samples(model, img_prefix, dataloader.batch_size)

            if iteration % save_freq == 0 and iteration > 0:
                save_checkpoint(checkpoint_dir, device, model, iteration=iteration)

            if iteration > max_iteration:
                break
            iteration += 1

        if iteration > max_iteration:
            break
        epoch += 1

    log_save_path = os.path.join(checkpoint_dir, "train-log.json")
    save_json(log, log_save_path)


if __name__ == '__main__':
    args = argparse_setup()
    main(args)
    """
