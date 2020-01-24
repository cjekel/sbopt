# rbfopt_py
WIP. Simple surrogate-based optimization in Python.

# What is Surrogate-based optimization?

Sometimes it is desirable to optimize a function that is really expensive. In these situations, we only want to evaluate our function at a new point if we have a strong belief that this point will improve on the function value. A surrogate model is fit to our expensive function, and then optimized to find a new point that will minimize our expensive function. The surrogate model is a function that is relatively cheap to evaluate, and is used in-place of the expensive function. This process is known as surrogate-based optimization.

The following [review paper](https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/20050186653.pdf) is a good place to start if you are new to surrogate-based optimization. 


```bibtex
@article{QUEIPO20051,
title = "Surrogate-based analysis and optimization",
journal = "Progress in Aerospace Sciences",
volume = "41",
number = "1",
pages = "1 - 28",
year = "2005",
issn = "0376-0421",
doi = "https://doi.org/10.1016/j.paerosci.2005.02.001",
url = "http://www.sciencedirect.com/science/article/pii/S0376042105000102",
author = "Nestor V. Queipo and Raphael T. Haftka and Wei Shyy and Tushar Goel and Rajkumar Vaidyanathan and P. Kevin Tucker",
abstract = "A major challenge to the successful full-scale development of modern aerospace systems is to address competing objectives such as improved performance, reduced costs, and enhanced safety. Accurate, high-fidelity models are typically time consuming and computationally expensive. Furthermore, informed decisions should be made with an understanding of the impact (global sensitivity) of the design variables on the different objectives. In this context, the so-called surrogate-based approach for analysis and optimization can play a very valuable role. The surrogates are constructed using data drawn from high-fidelity models, and provide fast approximations of the objectives and constraints at new design points, thereby making sensitivity and optimization studies feasible. This paper provides a comprehensive discussion of the fundamental issues that arise in surrogate-based analysis and optimization (SBAO), highlighting concepts, methods, techniques, as well as practical implications. The issues addressed include the selection of the loss function and regularization criteria for constructing the surrogates, design of experiments, surrogate selection and construction, sensitivity analysis, convergence, and optimization. The multi-objective optimal design of a liquid rocket injector is presented to highlight the state of the art and to help guide future efforts."
}
```