import os
import argparse
import pickle
from tqdm import tqdm
import PIL.Image
import numpy as np
import dnnlib
import dnnlib.tflib as tflib
import config
from encoder.generator_model import Generator
from encoder.perceptual_model import PerceptualModel
import tensorflow as tf
from scipy.stats import truncnorm

URL_FFHQ = 'https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ'  # karras2019stylegan-ffhq-1024x1024.pkl


def split_to_batches(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]

def combine(im_list, path):
    #images = map(PIL.Image.open, im_list)
    widths, heights = zip(*(i.size for i in im_list))

    total_width = sum(widths)
    max_height = max(heights)
    new_im = PIL.Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for im in im_list:
      new_im.paste(im, (x_offset,0))
      x_offset += im.size[0]

    #new_im.save(path)
    new_im.save(path, 'PNG')
    
def interpolation(dlatent1, dlatent2):
    dlatents = [dlatent2]
    x = 0
    for i in range(4):
      x += 0.2
      dlatents.append(dlatent1 * x + (1 - x) * dlatent2)
    dlatents.append(dlatent1)
    return dlatents

def main():
    parser = argparse.ArgumentParser(description='Find latent representation of reference images using perceptual loss')
    #parser.add_argument('src_dir', help='Directory with images for encoding')
    #parser.add_argument('generated_images_dir', help='Directory for storing generated images')
 
    # for now it's unclear if larger batch leads to better performance/quality
    parser.add_argument('--batch_size', default=1, help='Batch size for generator and perceptual model', type=int)

    # Perceptual model params
    parser.add_argument('--image_size', default=256, help='Size of images for perceptual model', type=int)
    parser.add_argument('--lr', default=1., help='Learning rate for perceptual model', type=float)
    parser.add_argument('--iterations', default=1500, help='Number of optimization steps for each batch', type=int)

    # Generator params
    parser.add_argument('--randomize_noise', default=False, help='Add noise to dlatents during optimization', type=bool)
    args, other_args = parser.parse_known_args()

    ref_images_a = [os.path.join('../stylegan_meta/data/anime/', x) for x in os.listdir('../stylegan_meta/data/anime')][20:]
    ref_images_d = [os.path.join('../stylegan_meta/dog', x) for x in os.listdir('../stylegan_meta/dog')]
    ref_images_c = [os.path.join('../stylegan_meta/cat', x) for x in os.listdir('../stylegan_meta/cat')]
    ref_images_f = [os.path.join('../stylegan_meta/flower', x) for x in os.listdir('../stylegan_meta/flower')]
    # iter: 32 name: ['image_0489']
    ref_images_a = list(filter(os.path.isfile, ref_images_a))
    ref_images_d = list(filter(os.path.isfile, ref_images_d))
    ref_images_c = list(filter(os.path.isfile, ref_images_c))
    ref_images_f = list(filter(os.path.isfile, ref_images_f))

    print(len(ref_images_c))

    #os.makedirs(args.generated_images_dir, exist_ok=True)
    #os.makedirs(args.dlatent_dir, exist_ok=True)
    #os.makedirs(args.random, exist_ok=True)
    #os.makedirs('interpolation/', exist_ok=True)

    os.makedirs('dlatent_dir_a/', exist_ok=True)
    os.makedirs('dlatent_dir_d/', exist_ok=True)
    os.makedirs('dlatent_dir_c/', exist_ok=True)
    os.makedirs('dlatent_dir_f/', exist_ok=True)
    os.makedirs('reconst/', exist_ok=True)
    os.makedirs('interpolation/', exist_ok=True)
    os.makedirs('random/', exist_ok=True)
    
    # Initialize generator and perceptual model
    tflib.init_tf()
    with dnnlib.util.open_url(URL_FFHQ, cache_dir='cache3/') as f:
        generator_network, discriminator_network, Gs_network = pickle.load(f)

    generator = Generator(Gs_network, args.batch_size, randomize_noise=args.randomize_noise)
    perceptual_model = PerceptualModel(args.image_size, layer=9, batch_size=args.batch_size)
    perceptual_model.build_perceptual_model(generator.generated_image)

    # Optimize (only) dlatents by minimizing perceptual loss between reference and generated images in feature space  
    itere = 1
    dlatent = []
    for images_batch in tqdm(split_to_batches(ref_images_d, args.batch_size), total=len(ref_images_d)//args.batch_size):
        # vars
        variable = []
        tvars = tf.trainable_variables()
        for var in tvars:
            if "StyleMod" in var.name or 'learnable_dlatents' in var.name:
                variable.append(var)
        names = [os.path.splitext(os.path.basename(x))[0] for x in images_batch]

        perceptual_model.set_reference_images(images_batch)
        loss = perceptual_model.optimize(variable, iterations=args.iterations, learning_rate=args.lr)
        print("iter: {0} name: {1} loss: {2}".format(itere, names, loss))

        # reconsruction
        generated_images = generator.generate_images()
        img = PIL.Image.fromarray(generated_images[0], 'RGB')
        img.save(os.path.join('reconst', f'{names[0]}.png'), 'PNG')
        
        generated_dlatents = generator.get_dlatents()
        dlatent.append(generated_dlatents)
        np.save(os.path.join('dlatent_dir_d', f'{names[0]}.npy'), generated_dlatents[0])
        
        # interpolation
        if len(dlatent) >= 2:
            #d1 = np.load('../../0a47b3000e84b81ddff6de16e9d76f7dfcb8eb6e_1.npy').reshape(1, 18, 512)
            #d2 = np.load('../../03ee9784beaacc12db1ee5e3aa4fa58890a70f7c.npy').reshape(1, 18, 512)
            d1 = dlatent[0]
            d2 = dlatent[1]
            d = interpolation(d1, d2)
            im_list = []
            for i in range(6):
                img = PIL.Image.fromarray(generator.generate_images(d[i])[0], 'RGB')
                im_list.append(img)
            combine(im_list, 'interpolation/{0}.png'.format(itere))

        # random
        w = truncnorm(-0.4, 0.4).rvs(18 * 512).astype("float32").reshape(1, 18, 512)
        rand_img = generator.generate_images(w)
        img = PIL.Image.fromarray(rand_img[0], 'RGB')
        img.save(os.path.join('random/', f'{names[0]}.png'), 'PNG')

        generator.reset_dlatents()
        itere += 1
    itere = 1

if __name__ == "__main__":
    main()
