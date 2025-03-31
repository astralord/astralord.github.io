---
layout: post
title: 'Attention-free Mechanisms'
date: 2025-02-01 10:00 +0800
categories: [Large Language Models, Alternative Architectures]
tags: [jax, ssm, s4, h3, mamba, hyena]
math: true
enable_d3: true
published: true
---

> Pre-ambula

## State Space Models (SSMs)

**State Space Models (SSMs)** are ...

### Transformer limitations

### Revisiting RNNs

RNN hidden state:

$$h_k = f(h_{k-1}, x_k)$$

RNN output:

$$y_k = g(h_k).$$

Usually, neural network layers include additional relation to input $x$ through skip connection, e.g. $y = g(h) + \mathbf{D}x$, but for the rest of this post we will omit the parameter $\mathbf{D}$, since this term is easy to compute.

### Fast CNNs

### LSSL

**Linear State Space Layer (LSSL)** is ...

$$f(h, x) = \mathbf{\bar{A}} h + \mathbf{\bar{B}} x$$

Let the initial hidden state be $h_{-1} = 0$, then the unroll of $k$-th hidden state yields

$$
\begin{aligned}
{h_0} &= \mathbf{\bar{B}}x_0 \\
{h_1} & = \mathbf{\bar{A}} {\mathbf{\bar{B}} x_0} + \mathbf{\bar{B}} x_1  \\
h_2 & = \mathbf{\bar{A}} {\mathbf{\bar{A}\bar{B}} x_0} + \mathbf{\bar{A}} {\mathbf{\bar{B}} x_1} + \mathbf{\bar{B}} x_2  \\
& \cdots \\
h_k &= \mathbf{\bar{A}}^k \mathbf{\bar{B}} x_{0} + \mathbf{\bar{A}}^{k-1}\mathbf{\bar{B}} x_{1} + \mathbf{\bar{A}}^{k-2}\mathbf{\bar{B}} x_{2} + \cdots + \mathbf{\bar{B}} x_k. \\
\end{aligned}
$$

If we set mapping $g$ also to be linear:

$$g(h) = \mathbf{\bar{C}}h,$$

we get

$$y_{k} = \mathbf{\bar{C}\bar{A}}^k \mathbf{\bar{B}} x_{0} + \mathbf{\bar{C}\bar{A}}^{k-1}\mathbf{\bar{B}} x_{1} + \cdots + \mathbf{\bar{C}\bar{B}} x_k.$$

Let's define an operator

$$
\mathcal{K}_k(\mathbf{A}, \mathbf{B}, \mathbf{C}) = \begin{pmatrix} 
\mathbf{CB} & \mathbf{CAB} & \cdots & \mathbf{CA}^{k-1}\mathbf{B}
\end{pmatrix} \in \mathbb{R}^k.
$$

Since matrices $\mathbf{\bar{A}}$, $\mathbf{\bar{B}}$, $\mathbf{\bar{C}}$ are all constant, we can precompute the coefficients up to a given maximum sequence length $L$ and store them together as a single giant tensor $\mathbf{\bar{K}}=\mathcal{K}_L(\mathbf{\bar{A}}, \mathbf{\bar{B}}, \mathbf{\bar{C}})$, called the **SSM convolution kernel**. Then we can represent $y$ as a simple convolution: **TODO: depthwise-separable**

$$
y = \mathbf{\bar{K}} \ast x.
$$

```python
class SSMLayer(nn.Module):
    N: int
    l_max: int
    decode: bool = False

    def setup(self):
        # SSM parameters
        self.A = self.param("A", lecun_normal(), (self.N, self.N))
        self.B = self.param("B", lecun_normal(), (self.N, 1))
        self.C = self.param("C", lecun_normal(), (1, self.N))
        self.D = self.param("D", nn.initializers.ones, (1,))

        # Step parameter
        self.log_step = self.param("log_step", log_step_initializer(), (1,))

        step = np.exp(self.log_step)
        self.ssm = discretize(self.A, self.B, self.C, step=step)
        self.K = K_conv(*self.ssm, self.l_max)

        # RNN cache for long sequences
        self.x_k_1 = self.variable("cache", "cache_x_k", np.zeros, (self.N,))

    def __call__(self, u):
        if not self.decode:
            # CNN Mode
            return causal_convolution(u, self.K) + self.D * u
        else:
            # RNN Mode
            x_k, y_s = scan_SSM(*self.ssm, u[:, np.newaxis], self.x_k_1.value)
            if self.is_mutable_collection("cache"):
                self.x_k_1.value = x_k
            return y_s.reshape(-1).real + self.D * u

```

|    | Convolution | Recurrence | Attention | S4 |
| -------- | ------- | ------- | ------- | ------- |
| Parameters  | $Ld$ | $d^2$ | $d^2$ | $d^2$
| Training | $\tilde{L}d(B+d)$ | $BLd^2$ | $B(L^2 d + Ld^2)$ | ...
| Inference | $Ld^2$ | $d^2$ | $L^2 d + Ld^2$ | ...

### Continuous-time representation

$$\frac{\partial }{\partial t} h(t) = \mathbf{A}h(t) + \mathbf{B}x(t)$$

To determine the relation between this continuous process and our discretized version, first let's find the derivative of $e^{-\mathbf{A}t} h(t)$:

$$
\begin{aligned}
\frac{\partial }{\partial t} \big (e^{-\mathbf{A}t} h(t) \big) &= -\mathbf{A} e^{-\mathbf{A}t} h(t) + e^{-\mathbf{A}t} \big(\mathbf{A}h(t) + \mathbf{B}x(t) \big) \\
&=e^{-\mathbf{A}t}\mathbf{B}x(t).
\end{aligned}$$

By assumption that $x(t)$ doesn't change on a sufficiently small discretization timestep $\Delta$ we can integrate both sides and derive that

$$
\begin{aligned}
e^{-\mathbf{A}(t+\Delta)} h(t + \Delta) - e^{-\mathbf{A}t} h(t) &= \int_t^{t+\Delta} e^{-\mathbf{A}s} \mathbf{B}x(s) ds & \\
&= \big(\mathbf{A}^{-1}  e^{-\mathbf{A}s} \big) \bigg \vert_{t+\Delta}^t \cdot \mathbf{B} x(t) & \color{Salmon}{\longleftarrow x(s) \text{ is const on } [t, t+\Delta]}  \\
&= \mathbf{A}^{-1} \big( e^{-\mathbf{A}t} - e^{-\mathbf{A}(t+\Delta)} \big) \cdot \mathbf{B}x(t).
\end{aligned}
$$

Finally, dividing both sides by $e^{-\mathbf{A}(t+\Delta)}$ and moving $-e^{\Delta\mathbf{A}} h(t)$ to the right side we get

$$
h(t + \Delta) = e^{\Delta\mathbf{A}}h(t) + \mathbf{A}^{-1} (e^{\Delta\mathbf{A}} - \mathbf{I})\cdot \mathbf{B} x(t)
$$

or

$$
h_{k + 1} = \mathbf{\bar{A}} h_k + \mathbf{\bar{B}} x_k
$$

with $h_k = h(\Delta \cdot k)$, $\mathbf{\bar{A}} = e^{\Delta\mathbf{A}}$ and $\mathbf{\bar{B}}=\mathbf{A}^{-1} (e^{\Delta\mathbf{A}} - \mathbf{I})\cdot \mathbf{B}$. Such way of discretization is called **Zero-Order Hold (ZOH)** method.[^DSC]

```python
def discretize(A, B, step):  
    A_bar = jnp.exp(step * A)
    A_inv = jnp.linalg.inv(A)
    I = jnp.eye(A.shape[0])
    B_bar = A_inv @ (A_bar - I) @ B
    return A_bar, B_bar
```

## Addressing Long-Range Dependencies

### Legendre polynomials

### HiPPO

**High-Order Polynomial Projection Operator (HiPPO)**

#### H3

### S4

**Structured State Space for Sequences (S4)** is ...

### Hyena

### Monarch ?

### S5 ?

### RWKV ?


## Selective SSM

### Content-awareness

Selective copying

Induction heads

### Parallel scan

### Hardware-aware algorithm

### Mamba

![S6 vs S4]({{'/assets/img/s4-s6.png'|relative_url}})

## State Space Duality

#### Mamba 2


[^DSC]: There are other ways to discretize $h(t)$ process. Standard numerical integration/differentiation rules can be described as a general formula: $$\frac{h(t+\Delta)-h(t)}{\Delta} = \mathbf{A} \big((1 - \alpha) h(t) + \alpha  h(t + \Delta)\big) + \mathbf{B}x(t),$$ where $\alpha \in [0, 1]$ is an interpolation coefficient. For $\alpha=0$ or $1$ we follow forward or backward Euler method respectively. And for $\alpha=\frac{1}{2}$ we use bilinear discretization leading to $$\begin{aligned}\mathbf{\bar{A}} &= \big(\mathbf{I} - \frac{\Delta}{2} \mathbf{A}\big)^{-1}\big(\mathbf{I} + \frac{\Delta}{2} \mathbf{A}\big), \\ \mathbf{\bar{B}} &= \big(\mathbf{I} - \frac{\Delta}{2} \mathbf{A}\big)^{-1} \Delta \mathbf{B}. \end{aligned}$$ Note, that if we use Taylor series approximation $e^{\pm \frac{\Delta}{2}\mathbf{A}} \approx \mathbf{I} \pm \frac{\Delta}{2} \mathbf{A}$ we get back to ZOH method.
