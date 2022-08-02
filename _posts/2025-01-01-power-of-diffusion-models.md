---
layout: post
title: 'Power of Diffusion Models'
date: 2022-01-01 11:00 +0800
categories: [Generative AI]
tags: [diffusion-models]
math: true
enable_d3: true
published: true
img_path: /assets/img/
---

> The year 2021 was a burst ...

![Bear in mind]({{'bear-in-mind.jpg'|relative_url}})

*Bear in mind, digital art. DALL·E-2 by OpenAI on Instagram.*

### Diffusion models

### DALL·E-2
#### CLIP

![CLIP]({{'clip-arch.png'|relative_url}})

*CLIP architecture*

```python
# image_encoder - ResNet or Vision Transformer
# text_encoder - CBOW or Text Transformer
# I[n, h, w, c] - minibatch of aligned images
# T[n, l] - minibatch of aligned texts
# W_i[d_i, d_e] - learned proj of image to embed
# W_t[d_t, d_e] - learned proj of text to embed
# t - learned temperature parameter

# extract feature representations of each modality
I_f = image_encoder(I) #[n, d_i]
T_f = text_encoder(T) #[n, d_t]

# joint multimodal embedding [n, d_e]
I_e = l2_normalize(np.dot(I_f, W_i), axis=1)
T_e = l2_normalize(np.dot(T_f, W_t), axis=1)

# scaled pairwise cosine similarities [n, n]
logits = np.dot(I_e, T_e.T) * np.exp(t)

# symmetric loss function
labels = np.arange(n)
loss_i = cross_entropy_loss(logits, labels, axis=0)
loss_t = cross_entropy_loss(logits, labels, axis=1)
loss = (loss_i + loss_t)/2
```

*JAX-like pseudocode for the core of an implementation of CLIP.*

### Guided diffusion

### Disco diffusion

### Midjourney