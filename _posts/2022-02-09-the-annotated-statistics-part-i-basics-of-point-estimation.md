---
layout: post
title: 'The Annotated Statistics. Part I: Basics of Point Estimation'
date: 2022-02-09 03:13 +0800
categories: [Statistics]
tags: [statistics, point-estimation, exponential-family, cramer-rao-inequality, fisher-information]
math: true
---

> This series of posts is a guidance to statistics for those who already have knowledge in probability theory and would like to become familiar with mathematical statistics. Part I focuses on point estimation of parameters for the most frequently used probability distributions.


## Intro

Imagine that you are a pharmaceutical company about to introduce a new drug into production. Prior to launch you need to carry out experiments to assess its quality depending on the dosage. Say you give this medicine to an animal, after which the animal is examined and checked whether it has recovered or not by taking a dose of $X$. You can think of the result as random variable $Y$ following Bernoulli distribution:

$$ Y \sim \operatorname{Bin}(1, p(X)), $$

where $p(X)$ is a probability of healing given dose $X$. 

Typically, several independent experiments $Y_1, \dots, Y_n$ with different doses $X_1, \dots, X_n$ are made, such that

$$ Y_i \sim \operatorname{Bin}(1, p(X_i)). $$ 
	
Our goal is to estimate function $p: [0, \infty) \rightarrow [0, 1]$. For example, we can simplify to parametric model

$$ p(x) = 1 - e^{-\vartheta x}, \quad \vartheta > 0. $$

Then estimating function $p(x)$ is equal to estimating parameter $\vartheta $.
<html><head>


<!-- Load require.js. Delete this if your page already loads require.js -->
<script src="https://cdnjs.cloudflare.com/ajax/libs/require.js/2.3.4/require.min.js" integrity="sha256-Ae2Vz/4ePdIu6ZyI/5ZGsYnb+m0JlOmKPjt6XZ9JJkA=" crossorigin="anonymous"></script>
<script src="https://unpkg.com/@jupyter-widgets/html-manager@*/dist/embed-amd.js" crossorigin="anonymous"></script>
<script type="application/vnd.jupyter.widget-state+json">
{
    "version_major": 2,
    "version_minor": 0,
    "state": {
        "0d4b43f1afa14e0590d176c4dbf0e923": {
            "model_name": "LayoutModel",
            "model_module": "@jupyter-widgets/base",
            "model_module_version": "1.2.0",
            "state": {}
        },
        "d4767b98c5c4447b81f6b9f3f3ae6dbe": {
            "model_name": "SliderStyleModel",
            "model_module": "@jupyter-widgets/controls",
            "model_module_version": "1.5.0",
            "state": {
                "description_width": ""
            }
        },
        "204bee1d0dd64721a07e0eda0378a7f8": {
            "model_name": "FloatSliderModel",
            "model_module": "@jupyter-widgets/controls",
            "model_module_version": "1.5.0",
            "state": {
                "description": "parameter",
                "layout": "IPY_MODEL_0d4b43f1afa14e0590d176c4dbf0e923",
                "max": 1,
                "readout": false,
                "step": 0.01,
                "style": "IPY_MODEL_d4767b98c5c4447b81f6b9f3f3ae6dbe"
            }
        },
        "844b195cb4024a4abeb06558954ac577": {
            "model_name": "LayoutModel",
            "model_module": "@jupyter-widgets/base",
            "model_module_version": "1.2.0",
            "state": {}
        },
        "b3c4d40a1cbf4de3b871835e8a65db23": {
            "model_name": "VBoxModel",
            "model_module": "@jupyter-widgets/controls",
            "model_module_version": "1.5.0",
            "state": {
                "_dom_classes": [
                    "widget-interact"
                ],
                "children": [
                    "IPY_MODEL_204bee1d0dd64721a07e0eda0378a7f8",
                    "IPY_MODEL_be1c11c21e944b76ab9b034c8456c388"
                ],
                "layout": "IPY_MODEL_844b195cb4024a4abeb06558954ac577"
            }
        },
        "6c372b5e78e74e4b907a100550b7f793": {
            "model_name": "LayoutModel",
            "model_module": "@jupyter-widgets/base",
            "model_module_version": "1.2.0",
            "state": {}
        },
        "be1c11c21e944b76ab9b034c8456c388": {
            "model_name": "OutputModel",
            "model_module": "@jupyter-widgets/output",
            "model_module_version": "1.0.0",
            "state": {
                "layout": "IPY_MODEL_6c372b5e78e74e4b907a100550b7f793",
                "outputs": [
                    {
                        "output_type": "display_data",
                        "data": {
                            "text/plain": "<Figure size 720x72 with 1 Axes>",
                            "image/png": "iVBORw0KGgoAAAANSUhEUgAAAmkAAABoCAYAAACqsmdeAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjMuMywgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/Il7ecAAAACXBIWXMAAAsTAAALEwEAmpwYAAALwklEQVR4nO3dz24UVxrG4fcbvEiYTYNglQWkfQX+M1eAuQObWWRtezQiirIJ4gqQ2UQINIrtdRbY3IGdG5h021dARyyyImN6E5JIbX2zqFNQbmOwwdQ59Pd7JAu73O0+L5jut86pqjZ3FwAAAMryt9wDAAAAwHGUNAAAgAJR0gAAAApESQMAACgQJQ0AAKBAlDQAAIACTeUewMdw5coVv379eu5hAAAAvFO/3//N3a+Ob5/Iknb9+nX1er3cwwAAAHgnM3v2pu0sdwIAABSIkgYAAFAgShoAAECBKGkAAAAFoqQBAAAUiJIGAABQIEoaAABAgShpAAAABaKkAQAAFIiSBgAAUCBKGgAAQIEoaQAAAAWipAEAABSIkgYAAFAgShoAAECBKGkAAAAFoqQBAAAUiJIGAABQIEoaAABAgShpAAAABZp62zfNbEfSjrvfH9v+QtKSu+++7wOnn/Gluw9PefuupL67X3rfx/wQo9FIGw8fSpJWvv5aU1NTb9w2iSJnl2Lnj5xdip2f7DGzS7Hzl5admbRTGI1G+uHBA/3e7+v3fl8/PHigv/7669i20WiUe6jnLnJ2KXb+yNml2PnJHjO7FDt/idnN3U/+5ic6kzY/P++9Xu99h3bMf77/Xr/3+/pqZkaS9OP+vn69eFFfvHx5ZNvf5+b072+/PbfHLUHk7FLs/JGzS7Hzkz1mdil2/pzZzazv7vPj289lJs3MumbWN7OnZrbW2L6eth3Zfob7L5rZCzPrS7pzHmP9UFMXLuirmRktfPaZvpqZ0dSFC7mH1JrI2aXY+SNnl2LnJ3vM7FLs/KVkP81M2rykg7FvdSXdrGfS0qzYDXffS/dZc/ddM1t4020aX3/p7sM33V9ST9KL+nHMbCX93NZn0uop0D/294/9Y40OD/Xj/r4+n5nRv775ZuLW6SNnl2Lnj5xdip2f7DGzS7Hz58z+ITNp99x9uvkhadj4wYuSenX5UlWwbkrS2HLorqrCNz6wk+5/S9Ju42e8dWnVzFbMrGdmvefPn58i1ulNTU1p+fZt/XrxovaePTvyvb1nz/TrxYtavn174n5hpdjZpdj5I2eXYucne8zsUuz8JWY/j+XOrqR6ubKvqmT9T5LMbMHMttPs2MIZ79+RNDjtINx9w93n3X3+6tWrHxDnuNFopM1Hj/TFy5eavXbtyPdmr13TFy9favPRo4k9kDJqdil2/sjZpdj5yR4zuxQ7f4nZz6OkDSU9cfe5xsf9dKD/uqRld78paess90/bu+cwvg+28fDhkenP0eGh/jsYaHR4+Grd+o/9/Ven6E6SyNml2PkjZ5di5yd7zOxS7PwlZj+PkrYlaTGVMplZx8wWVBWsYePszZNm0k66/5akBTObTbdbPIexfrB6XXr3zz/14/6+RoeHuYfUmsjZpdj5I2eXYucne8zsUuz8pWQ/l0twpCK1qWqJcqhq9mwvLV92VZ0EIEnb7r7R+Bn1iQMn3X8xbR+oOiZtwd3n3hXqY544IEmfz8xo+fZtbT56dGTbpB9IKcXKLsXOHzm7FDs/2WNml2Lnz5n9pBMH3lrSPlXnXdKk8q5C3KbI2aXY+SNnl2LnJ3vM7FLs/LmyU9IAAAAK9FEvZgsAAIDzRUkDAAAoECUNAACgQJQ0AACAAlHSAAAACkRJAwAAKBAlDQAAoECUNAAAgAJR0gAAAApESQMAACgQJQ0AAKBAlDQAAIACUdIAAAAKREkDAAAoECUNAACgQJQ0AACAAlHSAAAACkRJAwAAKBAlDQAAoECUNAAAgAKZu+cew7kzs+eSnn3kh7ki6beP/Bilipxdip0/cnYpdn6yxxU5f1vZr7n71fGNE1nS2mBmPXefzz2OHCJnl2Lnj5xdip2f7DGzS7Hz587OcicAAECBKGkAAAAFoqS9v43cA8gocnYpdv7I2aXY+ckeV+T8WbNzTBoAAECBmEkDAAAo0FTuAXxqzGxW0qa7z+UeS9tS9oX05T8kLbv7MN+I2mNmde6OquyP3X0v34jyMbN1d1/NPY62mNmapKeStiTdknTg7k/yjqpdZrYo6bKkgSS5+27eEbXDzLYV6Hmuycy6qp7vDyR1JW1E+ntI+RclDSVNu/udHONgJu0MGi/Us1kHkoGZdSTNu/t9d78v6bGkn/KOqlXbknrpxflnSZuZx5NFKuoruceRwbqkXyR1gha0rrtvqCppa5mH1KZFSS/MzBsf3+UeVEsW3X3D3Z+k5/y7uQfUsp30erch6XHaWWsdJe0M3H036uyJpHlJzT2JXUmzqbxFMNfYi7ysau8yoq6qPctIfnZ3c/dL6cUqmrU6t7sPoqwipJmUm+nf3tzdJK0G+h24OfZ1J8cgckg7JoP66/S6n2XnlJKGU0nLG0uNTd20fZhlQC1z90HjyyXFmk2QVD1xRZtFakqziKGkzMPG55EcNJd10wv3VsbxtM7Mdsysk1aRtnOPJ7NOKu6toqTh1MZmEf8pKcoepaRqzzotdWxHOSanlp6cBu+84WTqphepgZmtBSsrXUkH9cyCmX3XOOxjojV3QNOKweUoO6XJkqpVg18kzQZ7zttTmoiQjuygdNoeCJfgeA9m5mnqO6T0hLXt7uPT4RMvZV9TdbxCmFml5iyamb1w90u5x5RDKqs77j6deyxtMLMVSev18136/f8l2r9/Oh7pXqSS1ijjXVXHZN4JtNSrtEM+VDV7Oq9qJnFubFXl44+DknZ2lDSr/8MOc48lh/TktSPpUoS/g5S3V2eNVtLMrDM2qxLm/3+aQbvbPA7NzFzV2W5hZlbNrB/lWDzp9ZmNdSlLX/clfRnhOa/WmEEd5Pp/z3InziTtXdxx92GUkwbMbMHM+o1NvfTn5RzjyeSWma2kmZVO+rz14zPalgpqpLOYx73pRKmhAp04k34HwuRNFlSdHCbp1TG5G4r1nCd3H6aCNqvG30ebuE4aTi3tVT9p7EktSIqw5Heg6pIjtXlJgygzCePHoqTrpEV5m5iepHv1F/X/gXzDaVd6gTqoZxPrYxMjzaaouuTSMPcgWrar6vIjR0p6lOc86diKwaqOXt2gNZS0M0h7VLPp8/q4pBAHU6Yn5+30eb15oAAvWO6+l04aqE/BntPx09MnXpo5XUmfr6k6Vmmin7RTMRk0jk+Zdveld9xt0ixJumtmTyVNS7qReTxtG6q6NmIYqZzXv/cDVTNo65mH1bblxkWct3Ndfotj0gAAAArEMWkAAAAFoqQBAAAUiJIGAABQIEoaAABAgShpAAAABaKkAQAAFIiSBmDimNmOmb0ws376eJqu7db2OL4zM0/j6aRt3TS27WBv1g7gjChpACbVHXefSx/Tqt7OaqfNAaT3Pnyi41fpX3b3pVwXyATwaaCkAQjB3VclXW68c0Rb7un1OzV0JM26+8S/UweAD0dJAxDJuqr34ZP0aumxXg7dbixJdtIS5dP0sfKG259q+TTNltVvsXOLggbgtChpACLp6fX773Yk9SUtpeXQx5J+Sre7JWnP3afT9+r36O2rWqqcljSb3s/3NNYk3Q30xvQAzgElDUBUtyRt1W8Sn2a4Oo2D+Rfrz9MbTi9K6jWOI1uTdPMMj9fhRAEAZzGVewAA0KKupLpkTUsajn1/IKnr7htmNifpJzOTpBvpvl0z6zdu//hdD5iWSrckzalaal19+z0AoMJMGoBI7up1sXqqqng1dVUVNbn7qrtfUnXg/6aqQvekccboXDp780Rp9m0rndm5rnQCAQCcBiUNQAhmti69uiyGVM1uLdRLkGnGa+jue2a2YGZ1gduV1Em3X6y3p5MLTjwmLRW0V5feSMukexnOLgXwiWK5E8CkumNmzaXFXXefq79w96GZ3ZC0mU4iGKha1pSqUvbqbE9Jq+n2S43tQ0nL4w+aSt9dSQuqljjr7d30c++Y2QFneQJ4F3P33GMAAADAGJY7AQAACkRJAwAAKBAlDQAAoECUNAAAgAJR0gAAAApESQMAACgQJQ0AAKBAlDQAAIAC/R9q9V4WJEC6lAAAAABJRU5ErkJggg==\n"
                        },
                        "metadata": {
                            "needs_background": "light"
                        }
                    }
                ]
            }
        },
        "088b24e868fc45889ef79d6dec30895b": {
            "model_name": "LayoutModel",
            "model_module": "@jupyter-widgets/base",
            "model_module_version": "1.2.0",
            "state": {}
        },
        "ac7b8e67b8064adb8dcabf62d5383e1e": {
            "model_name": "SliderStyleModel",
            "model_module": "@jupyter-widgets/controls",
            "model_module_version": "1.5.0",
            "state": {
                "description_width": ""
            }
        },
        "e46b973d3d5948529d5993d9ee936990": {
            "model_name": "FloatSliderModel",
            "model_module": "@jupyter-widgets/controls",
            "model_module_version": "1.5.0",
            "state": {
                "description": "parameter",
                "layout": "IPY_MODEL_088b24e868fc45889ef79d6dec30895b",
                "max": 1,
                "readout": false,
                "step": 0.01,
                "style": "IPY_MODEL_ac7b8e67b8064adb8dcabf62d5383e1e"
            }
        },
        "addf1444e8e44b45936acd309d098e5a": {
            "model_name": "LayoutModel",
            "model_module": "@jupyter-widgets/base",
            "model_module_version": "1.2.0",
            "state": {}
        },
        "b9411d9eac8e4d43831354d21b08d134": {
            "model_name": "VBoxModel",
            "model_module": "@jupyter-widgets/controls",
            "model_module_version": "1.5.0",
            "state": {
                "_dom_classes": [
                    "widget-interact"
                ],
                "children": [
                    "IPY_MODEL_e46b973d3d5948529d5993d9ee936990",
                    "IPY_MODEL_9cac8c48e05b450e9335640a13f0a6f6"
                ],
                "layout": "IPY_MODEL_addf1444e8e44b45936acd309d098e5a"
            }
        },
        "8c5c7bb418d64404be429b69752dcbf2": {
            "model_name": "LayoutModel",
            "model_module": "@jupyter-widgets/base",
            "model_module_version": "1.2.0",
            "state": {}
        },
        "9cac8c48e05b450e9335640a13f0a6f6": {
            "model_name": "OutputModel",
            "model_module": "@jupyter-widgets/output",
            "model_module_version": "1.0.0",
            "state": {
                "layout": "IPY_MODEL_8c5c7bb418d64404be429b69752dcbf2",
                "outputs": [
                    {
                        "output_type": "display_data",
                        "data": {
                            "text/plain": "<Figure size 720x72 with 1 Axes>",
                            "image/png": "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\n"
                        },
                        "metadata": {
                            "needs_background": "light"
                        }
                    }
                ]
            }
        }
    }
}
</script>
</head>
<body>

<script type="application/vnd.jupyter.widget-view+json">
{
    "version_major": 2,
    "version_minor": 0,
    "model_id": "b9411d9eac8e4d43831354d21b08d134"
}
</script>

</body>
</html>


### Notations

Here is a list of notations to help you read through equations in the post easily.

| Symbol | Meaning |
| ----------------------------- | ------------- |
| $$(\Omega, \mathcal{A}, \mathbb{P})$$ | **Probability space**: triplet of <br> $\cdot$ set of all possible outcomes $\Omega$, <br> $\cdot$ $\sigma$-algebra (event space) $\mathcal{A}$, <br> $\cdot$ probability measure $\mathbb{P}$. |
| $$ (\mathcal{X}, \mathcal{B}) $$ | Measurable space, defined by set $\mathcal{X}$ and $\sigma$-algebra $\mathcal{B}$. |
| $$ X: (\Omega, \mathcal{A}, \mathbb{P}) \rightarrow (\mathcal{X}, \mathcal{B}) $$ | Random variable. Recall that random variable is a function defined on a set of possible outcomes $\Omega$ and it maps to measurable space $\mathcal {X}$. <br> If we define measure $P(B) = \mathbb{P}(X^{-1}(B))$, $B \in \mathcal{B}$, then triplet $(\mathcal{X}, \mathcal{B}, P)$ is also a probability space and $\mathcal{X}$  is called **sample space**. |
| $$ x = X(\omega) $$ | Sample, element of $\mathcal {X}$. |
| $$ \Theta $$ | **Parametric space**, $\mid \Theta \mid \geq 2$. |
| $$ \mathcal{P} = \{ P_\vartheta \mid \vartheta \in \Theta \} $$ | Family of probability measures on $(\mathcal{X}, \mathcal{B})$, where $P_\vartheta \neq P_{\vartheta'} \ \forall \vartheta \neq \vartheta'$. |

The idea is that we are interested in the true distribution $P \in \mathcal{P}$ of random variable $X: \Omega \rightarrow \mathcal{X}$. On the basis of $x=X(\omega)$ we make a decision about the unknown $P$. By identifying family $\mathcal{P}$ with the parameter space $\Theta$, a decision for $P$ is equivalent to a decision for $\vartheta$. In our example above:

$$ Y_i \sim \operatorname{Bin}(1, 1 - e^{-\vartheta X_i}) = P_i^\vartheta. $$ 

Formally,

$$ \mathcal{X}=\{ 0, 1 \}^n, \quad \mathcal{B}=\mathcal{P(X)}, \quad \mathcal{P}=\{\otimes_{i=1}^nP_i^{\vartheta} \mid \vartheta>0 \}, \quad \Theta=\left[0, \infty\right). $$


### Uniformly best estimator



## UMVU estimator 

* Let $g$ be an estimation of $\gamma$, then

  $$ B_\theta(g) = \mathbb{E}_\theta[g(X)] - \gamma(\theta) $$

  is called **bias** of $g$. Estimation $g$ is called **unbiased** if 
  
  $$ B_\theta(g) = 0 \quad \forall \theta \in \Theta.$$

* Estimator $\tilde{g}$ is called **uniformly minimum variance unbiased (UMVU)** if

  $$ \tilde{g} \in \mathcal{E}_\gamma = \{g| B_\theta(g) \} $$

and

  $$\operatorname{Var}_\theta(\tilde{g}(X)) = \mathbb{E}_\theta[(\tilde{g}(X) - \gamma(\theta))^2] = \inf_{g \in \mathcal{E}_\gamma} \operatorname{Var}(g(X)).$$


## Efficient estimator

...

