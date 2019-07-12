import os
import argparse
import tensorflow as tf
from encoder.perceptual_model import LossModel
import pickle
import dnnlib
import dnnlib.tflib as tflib
import config
import PIL.Image
import numpy as np
from encoder.generatorModel import Generator
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
# loss_model.build_loss_model()

# Optimize (only) dlatents by minimizing perceptual loss between reference and generated images in feature space
# vars
variable = []
    tvars = tf.trainable_variables()
    for var in tvars:
        if "StyleMod" in var.name or 'learnable_dlatents' in var.name:
            variable.append(var)

for images_batch in tqdm(split_to_batches(ref_images, args.batch_size), total=len(ref_images) // args.batch_size):
    names = [os.path.splitext(os.path.basename(x))[0] for x in images_batch]
    
    # Generate images from found dlatents and save them
    generated_images = generator.generate_images()
        op = loss_model.optimize(generated_images, images_batch, variable, iterations=args.iterations, learning_rate=args.lr)
        
        pbar = tqdm(op, leave=False, total=args.iterations)
        for loss in pbar:
            pbar.set_description(' '.join(names)+' Loss: %.2f' % loss)
        print(' '.join(names), ' loss:', loss)
        
        # reconstruction ./gen
        generated_images = generator.generate_images()
        # random ./dgen
        random_mages = generator.generate_images(tf.random_normal(generator.dlatent_variable.shape))
        
        # generated_dlatents = generator.get_dlatents()
        for rec, ran, img_name in zip(generated_images, random_mages, names):
            rec_img = PIL.Image.fromarray(rec, 'RGB')
            ran_img = PIL.Image.fromarray(ran, 'RGB')
            rec_img.save(os.path.join('./gen', f'{img_name}.png'), 'PNG')
            ran_img.save(os.path.join('./dgen', f'{img_name}.png'), 'PNG')

# generator.reset_dlatents()


if __name__ == "__main__":
    main()





"""
    import os
    import argparse
    import tensorflow as tf
    from encoder.perceptual_model import LossModel
    import pickle
    import dnnlib
    import dnnlib.tflib as tflib
    import config
    import PIL.Image
    import numpy as np
    from encoder.generator_model import Generator
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
    parser.add_argument('--image_size', default=1024, help='Size of images for perceptual model', type=int)
    parser.add_argument('--lr', default=1., help='Learning rate for perceptual model', type=float)
    parser.add_argument('--iterations', default=1000, help='Number of optimization steps for each batch', type=int)
    
    # Generator params
    parser.add_argument('--randomize_noise', default=False, help='Add noise to dlatents during optimization', type=bool)
    args, other_args = parser.parse_known_args()
    
    # ----------------------------------------------------------------------
    ref_images = [os.path.join('data/anime/', x) for x in os.listdir('data/anime/')]
    ref_images = list(filter(os.path.isfile, ref_images))
    
    if len(ref_images) == 0:
    raise Exception('%s is empty' % 'data/anime/')
    print('dataset size: %d' % len(ref_images))
    
    os.makedirs('gen/', exist_ok=True)
    os.makedirs('dgen/', exist_ok=True)
    
    # ----------------------------------------------------------------------
    # Initialize generator and perceptual model
    tflib.init_tf()
    with dnnlib.util.open_url('https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ', cache_dir=config.cache_dir) as f:
    generator_network, discriminator_network, Gs_network = pickle.load(f)
    
    generator = Generator(Gs_network, args.batch_size, randomize_noise=args.randomize_noise)
    loss_model = LossModel(args.image_size, layer=9, batch_size=args.batch_size)
    loss_model.build_loss_model(generator.generated_image)
    
    # Optimize loss wrt variables
    for images_batch in tqdm(split_to_batches(ref_images, args.batch_size), total=len(ref_images) // args.batch_size):
    names = [os.path.splitext(os.path.basename(x))[0] for x in images_batch]
    loss_model.set_reference_images(images_batch)
    # latent variable
    variable = [generator.dlatent_variable]
    # batch sta
    tvars = tf.trainable_variables()
    print("==============")
    for var in tvars:
    if "StyleMod" in var.name:
    variable.append(var)
    if "learnable_dlatents" in var.name:
    print(var.name)
    print("==============")
    op = loss_model.optimize(variable, iterations=args.iterations, learning_rate=args.lr)
    # Generate images from found dlatents and save them
    generated_images = generator.generate_images()
    generated_dlatents = generator.get_dlatents()
    for img_array, dlatent, img_name in zip(generated_images, generated_dlatents, names):
    img = PIL.Image.fromarray(img_array, 'RGB')
    # random
    img.save(os.path.join('gen', f'{img_name}.png'), 'PNG')
    # reconstruction
    img.save(os.path.join('dgen', f'{img_name}.png'), 'PNG')
    # interpolation
    
    generator.reset_dlatents()
    """
