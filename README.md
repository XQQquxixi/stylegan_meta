# Introduction
Few-shot learning has been studied in the context of classification and achieved great success. In the meanwhile, GAN also had many breakthroughs with the proposal of StyleGAN etc. Here we are interested in Few-shot generation, specifically with GAN, so given a small amount of data in some class, which the model has never seen before, we want it to generate images  from the same class.

# Related Work
We believe Style-Based Generator is the state of the art, so we build our model on top of it. One interesting paper about it is ***Image2StyleGAN*** by Rameen Abdal et al., they found that *StyleGAN-ffhq*
is so expressive that it is potentially capable of generating any images, although only human faces' embedding is meaningful (i.e. interpolation is smooth and continuous). Another work that's more related to our idea is ***Image Generation from Small Datasets via Batch Statistics Adaptation***, the model can generate anime or human faces given as few as 25 images. Other people have also done interesting work such as ***Few-Shot Unsupervised Image-to-Image Translation*** and ***FIGR*** etc..

# Our Approach
Inspired by the work above we propose an algorithm that \textbf{makes embedding of any class meaningful by updating the AdaIn mappings while enforce the interpolation constraints explicitly}:\\
Let $\mathbb{A}_i$(w) be the $i$th Adain mapping, i.e. the mapping from $w$ to the bath normalization statistics for layer $i$ of the style-based generator network. Suppose we are given a set of images $(x_1, \dots, x_k)$ in a class. We move AdaIn mappings by finding a set of $z_i$ and $A_i$ such that
\begin{align*}
    L = \sum_{i, j} \gamma\|G(z_i)-z_i\|_1+\tau L_{per}(G(z_i), z_i) + \eta L(G(\alpha z_i + \beta z_j), R(\alpha) \cdot x_i + R(\alpha) \cdot x_j)
\end{align*}
where $0 \leq \alpha, \beta \leq 1, \alpha + \beta = 1$ is minimized.\\
To enforce the third term i.e. the interpolation loss, there are a few ways we can do this. 
\begin{enumerate}
    \item Train a k-shot discriminator that checks the interpolation points and makes sure they look like the $k$ points provided for few shot learning. 
    \item Enforce the middle image has the content of the leftmost image and the style of the rightmost image
    \item Use the L1 loss scaled by the distance of the interpolated variable from the endpoints. Suppose that $\gamma_p = \alpha z_i + \beta z_j$. Then one possible loss is $\alpha \cdot \frac{\| x_i - G(\gamma_p) \|_1}{\|x_i \|_2^2+ \|x_j\|_2^2} + \beta \cdot \frac{\| x_j - G(\gamma_p) \|_1}{\|x_i \|_2^2+ \|x_j\|_2^2}$
\end{enumerate}

# Results
These results are form implementation of the (3) in above, with simply $L_per$ and $L_1$, using 50 training examples.
interpolation results:\\
\includegraphics[scale=0.06]{22.png}
\includegraphics[scale=0.06]{43.png}
\includegraphics[scale=0.06]{59.png}
\includegraphics[scale=0.06]{82.png}
\includegraphics[scale=0.06]{6.png}
\includegraphics[scale=0.06]{58.png}
\includegraphics[scale=0.06]{50.png}
interpolation without interpolation loss:\\
\includegraphics[scale=0.06]{7.png}
\includegraphics[scale=0.06]{k.png}\\
interpolation sampling results (depth one, too little depth so not very meaningful):\\
\includegraphics[scale=0.1]{80_inter.png}
\includegraphics[scale=0.1]{89_inter.png}\\
random sampling results:\\
\includegraphics[scale=0.1]{81_w.png}
\includegraphics[scale=0.1]{82_w.png}
\includegraphics[scale=0.1]{86_w.png}
\includegraphics[scale=0.1]{305_w.png}\\

# Try the Sample Code
```
python ./train
```
