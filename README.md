# Predictive variational inference for flexible regression models
Code for the paper "Predictive variational inference for flexible regression models" by Lucas Kock, Scott A. Sisson, G. S. Rodrigues, and David J. Nott

A conventional Bayesian approach to prediction uses the posterior distribution to integrate
out parameters in a density for unobserved data conditional on the observed data and
parameters. When the true posterior is intractable, it is replaced by an approximation;
here we focus on variational approximations. Recent work has explored methods that learn
posteriors optimized for predictive accuracy under a chosen scoring rule, while regulariz-
ing toward the prior or conventional posterior. Our work builds on an existing predictive
variational inference (PVI) framework that improves prediction, but also diagnoses model
deficiencies through implicit model expansion. In models where the sampling density de-
pends on the parameters through a linear predictor, we improve the interpretability of
existing PVI methods as a diagnostic tool. This is achieved by adopting PVI posteriors
of Gaussian mixture form (GM-PVI) and establishing connections with plug-in predic-
tion for mixture-of-experts models. We make three main contributions. First, we show
that GM-PVI prediction is equivalent to plug-in prediction for certain mixture-of-experts
models with covariate-independent weights in generalized linear models and hierarchical
extensions of them. Second, we extend standard PVI by allowing GM-PVI posteriors to
vary with the prediction covariate and in this case an equivalence to plug-in prediction
for mixtures of experts with covariate-dependent weights is established. Third, we demon-
strate the diagnostic value of this approach across several examples, including generalized
linear models, linear mixed models, and latent Gaussian process models, demonstrating
how the parameters of the original model must vary across the covariate space to achieve
improvements in prediction.

## Minimal working example
A minimal working example on simulated data is given in minimal_working_example.py. This is a simple example to understand how VGM-PVI can be applied to new model specifications. 

## VGM-PVI
vgm_pvi.py contains code to train VGM-PVI on generalised linear models (GLMs). Extensions to hierarchical models and Gaussian process models are given in the respective sub-folders. The VGM_PVI class has a modular structure, in which the likelihood model and the prior need to be specified. It takes the following inputs:

     - p: dimension of the linear predictor / dimension of the parameter vector theta
     - k_init: initial number of mixture components
     - model: name of the underlying base model:
                - 'Gaussian': for a Gaussian regression model with fixed variance
                - 'Gaussian_var': for a Gaussian regression model with unknown variance, the last entry in theta corresponds to the variance in this model
                - 'Poisson': a Poisson GLM with log-link
                - 'Bernoulli': logistic regression
     - prior: prior for theta:
                - 'Gaussian': joint Gaussian prior  N(0, τ^2 I). tau needs to be specified in the dictionary additional_parameters
                -  'multGaussian': a multivariate Gaussian prior N(0,Omega). Omega needs to be specified in the dictionary additional_parameters.
     - diagonal_covariance: True if the mixture covariance matrices are restricted to be diagonal (default: False)
     - additional_parameters: Dictionary of additional hyperparameters for the prior and/or likelihood model

To extend this code to custom likelihood and prior specifications, respective analytic formulas for the ELBO need to be added. Further comments are given directly within the code and it is recommended to run minimal_working_example.py first to familiarize yourself with the set-up.

## Reproduction of results from the paper
To reproduce the experiments from the paper you need to run the respective .py file. The code is tested with Python 3.12.2. 

  a) simulation_binomial.py reproduces the simulation study presented in Section 4.1 of the main paper.

  b) Gamma_telescope_data.py reproduces the logistic regression example presented in Section 4.2 of the main paper.

  c) Aids_case_counts.py reproduces the poisson regression example presented in Section 4.3 of the main paper.

  d) Code for the extension of VGM-PVI to hierarchical models in the context of the water temperature example discussed in Section 5 of the main paper is given in the sub-folder hierarchical models

  e) Code for the extension of VGM-PVI to latent Gaussian process models in the context of the example discussed in Section 6 of the main paper is given in the sub-folder latent_Gaussian_processes

## Citation

If you use this work, please cite our paper.

