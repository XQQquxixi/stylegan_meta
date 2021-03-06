# Introduction
Few-shot learning has been studied in the context of classification and achieved great success. In the meanwhile, GAN also had many breakthroughs with the proposal of StyleGAN etc. Here we are interested in Few-shot generation, specifically with GAN, so given a small amount of data in some class, which the model has never seen before, we want it to generate images from the same class.

# Related Work
We believe Style-Based Generator is the state of the art, so we build our model on top of it. One interesting paper about it is ***Image2StyleGAN*** by Rameen Abdal et al., they found that *StyleGAN-ffhq*
is so expressive that it is potentially capable of generating any images, although only human faces' embedding is meaningful (i.e. interpolation is smooth and continuous).

Another work that's more related to our idea is ***Image Generation from Small Datasets via Batch Statistics Adaptation***, the model can generate anime or human faces given as few as 25 images. The core idea of this paper is that convolution can be seen as a combination of filters, by changing scale and shift we can control filter selection. Other people have also done interesting work such as ***Few-Shot Unsupervised Image-to-Image Translation*** and ***FIGR*** etc..

# Our Approach
Inspired by the work above we propose an algorithm that ***makes embedding of any class meaningful by updating the AdaIn mappings while enforce the interpolation constraints explicitly***. AdaIn is defined as follows:

![](https://github.com/XQQquxixi/stylegan_meta/blob/master/images/AdaIn.png)

We choose to only update AdaIn mappings because it changes scale and shift, and thus controls style as it presents as filter selection. Let A_l(w) be the l-th Adain mapping, i.e. the mapping from w to the batch normalization statistics for layer l of the style-based generator network, and suppose we are given a set of images (x_1, ..., x_k) in a class. We move AdaIn mappings by finding a set of z_i and A_l such that

 ![](https://github.com/XQQquxixi/stylegan_meta/blob/master/images/CodeCogsEqn.png)

where 0 <= \alpha, \beta <= 1, \alpha + \beta = 1 is minimized.
To enforce the third term L' i.e. the interpolation loss, there are a few ways we can do this. 

1. Train a k-shot discriminator that checks the interpolation points and makes sure they look like the $k$ points provided for few shot learning. 

2. Enforce the middle image has the content of the leftmost image and the style of the rightmost image

3. Use the L1 loss scaled by the distance of the interpolated variable from the endpoints. Suppose that ![](https://github.com/XQQquxixi/stylegan_meta/blob/master/images/CodeCogsEqn-2.png). Then one possible loss is ![](https://github.com/XQQquxixi/stylegan_meta/blob/master/images/CodeCogsEqn-3.png)
# Results
These results are from the implementation of (3) above using 50 training examples. In addition to $L_1$, we added $L_per$.
### interpolation results:
![alt text](https://github.com/XQQquxixi/stylegan_meta/blob/master/images/22.png)
![alt text](https://github.com/XQQquxixi/stylegan_meta/blob/master/images/59.png)
![alt text](https://github.com/XQQquxixi/stylegan_meta/blob/master/images/82.png)
![alt text](https://github.com/XQQquxixi/stylegan_meta/blob/master/images/50.png)
![alt text](https://github.com/XQQquxixi/stylegan_meta/blob/master/images/75.png)
![alt text](https://github.com/XQQquxixi/stylegan_meta/blob/master/images/70.png)
### The importance of interpolation loss:
First two rows are interpolations without interpolation loss defined above, last two rows are with:
![alt text](https://github.com/XQQquxixi/stylegan_meta/blob/master/images/7.png)
![alt text](https://github.com/XQQquxixi/stylegan_meta/blob/master/images/without_inter.png)
![alt text](https://github.com/XQQquxixi/stylegan_meta/blob/master/images/6.png)
![alt text](https://github.com/XQQquxixi/stylegan_meta/blob/master/images/43.png)
### The importance of updating AdaIn:
This is interpolation result without updating AdaIn parameters, first and last pictures represents reconstructions of two pictures, getting from Style-GAN trained from human face dataset, similar to what ***Image2StyleGAN*** has shown, the reconstruction is super good but the embedding of flower picture is not meaningful, actually the model "overfits" to human faces:
![](https://github.com/XQQquxixi/stylegan_meta/blob/master/images/without_addin.png)
This is interpolation result from our model, now the interpolation is quite smooth:
![alt text](https://github.com/XQQquxixi/stylegan_meta/blob/master/images/13.png)

### Interpolation sampling results (depth one, first two) and random sampling results (following three):
<img src="https://github.com/XQQquxixi/stylegan_meta/blob/master/images/80_inter.png" width="150" height="150">  <img src="https://github.com/XQQquxixi/stylegan_meta/blob/master/images/89_inter.png" width="150" height="150"> <img src="https://github.com/XQQquxixi/stylegan_meta/blob/master/images/159_w0.png" width="150" height="150"> <img src="https://github.com/XQQquxixi/stylegan_meta/blob/master/images/81_w.png" width="150" height="150"> <img src="https://github.com/XQQquxixi/stylegan_meta/blob/master/images/86_w.png" width="150" height="150"> 
Here, interpolation sampling means image is generated from latent code that is interpolated from training images latent codes. Random sampling means a random latent code from latent space.
### Random results for flowers:
<img src="https://github.com/XQQquxixi/stylegan_meta/blob/master/images/210_w1.png" width="150" height="150">  <img src="https://github.com/XQQquxixi/stylegan_meta/blob/master/images/211_w1.png" width="150" height="150"> <img src="https://github.com/XQQquxixi/stylegan_meta/blob/master/images/497_w1.png" width="150" height="150"> <img src="https://github.com/XQQquxixi/stylegan_meta/blob/master/images/35_0.png" width="150" height="150">  <img src="https://github.com/XQQquxixi/stylegan_meta/blob/master/images/216_w3.png" width="150" height="150"> 



# Try the Sample Code
```
python ./train
```
