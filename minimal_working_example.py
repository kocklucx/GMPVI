import torch
import numpy as np
import matplotlib.pyplot as plt
from vgm_pvi import VGM_PVI

# simulate data
n = 2000 # number of observations
sigma2 = 0.5 # known variance in the Gaussian likelihood
tau2 = 100 # pior variance
covariable = 4*torch.rand((n,1),dtype=torch.float64)-2
y = covariable.pow(3)+torch.tensor(sigma2).sqrt()*torch.randn((n,1))

# set up base model
x = torch.cat([torch.ones((n,1)),covariable],axis=1) # design matrix

# train VGM-PVI
pvi = VGM_PVI(p=len(x[0]),k_init=5,model='Gaussian',prior='Gaussian',additional_parameters={'tau2': tau2, 'sigma2': sigma2})
pvi.train(y=y,x=x,beta=0.05)

# true posterior mean
cov_posterior = torch.linalg.inv(1/tau2+1/sigma2*x.T@x)
mu_posterior =  1/sigma2 * cov_posterior@x.T@y

# summarize results
pos = torch.linspace(-2,2,500,dtype=torch.float64)
x_plot = torch.vstack([torch.ones(500),pos]).T

predicted_mean_posterior = x_plot@mu_posterior 

w,m,s = pvi.gmm_parameters(x_plot)
predicted_mean_pvi = (w*(x_plot@m.T)).sum(axis=1)

# plot results
fig, ax = plt.subplots()
ax.scatter(covariable,y,s=3,color='black')
ax.plot(pos,predicted_mean_posterior,label='posterior')
ax.plot(pos,predicted_mean_pvi,label='VGM-PVI')
ax.legend()
plt.show()