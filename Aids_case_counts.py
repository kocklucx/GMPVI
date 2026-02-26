import torch
import numpy as np
import matplotlib.pyplot as plt
import pymc as pm
from vgm_pvi import VGM_PVI
from copy import deepcopy
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import LogExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models.transforms.outcome import Standardize


def learn_vgm_pvi(beta):
    # learns VGM_PVI for a given penalty parameter beta
    pvi = VGM_PVI(p=len(x[0]),k_init=10,model='Poisson',prior='Gaussian',additional_parameters={'tau2': 5.0})
    pvi.train(y=y,x=x,beta=beta,n_iter=50000)
    score = pvi.elpd_waic(y,x)
    print(beta,score.item())
    return pvi, score

# load data and specify design matrix
y = torch.tensor([2,6,10,8,12,9,28,28,35,32,46,47,50,61,99,95,150,143,197,159,204,168,196,194,210,180,277,181,327,276,365]).unsqueeze(1)#,300,356,304,307,386,331,358,415,374,412,358,416,414,496
n = len(y)
t = torch.linspace(0,1,n)
t2 = t.pow(2)
intercept = torch.ones(n)
q1 = torch.zeros(n)
q1[::4]=1
q2 = torch.zeros(n)
q2[1::4]=1
q3 = torch.zeros(n)
q3[2::4]=1
x = torch.stack([intercept,t,t2,q1,q2,q3]).T#

# sample from conventional posterior
x_np = np.asarray(x)
y_np = np.asarray(y).flatten()
with pm.Model() as poisson_model:
    theta = pm.Normal('theta', mu=0, sigma=np.sqrt(5), shape=x.shape[1])
    linear_predictor = pm.math.dot(x_np,theta)
    likelihood = pm.Poisson('likelihood', mu=pm.math.exp(linear_predictor), observed=y_np)
    trace = pm.sample(5000, tune=1000, cores=1, chains=1) #chains=4
posterior_samples = np.concatenate(trace.posterior["theta"].values,axis=0)

# initalize Bayesian optimization and find the optimal VGM-PVI
bounds = torch.stack([torch.zeros(1), torch.ones(1)]).to(torch.double)
beta_start = 0.4
candidates = [beta_start*torch.ones(1)]
pvi, score = learn_vgm_pvi(100**beta_start-0.99)
scores = [score]
optimal_score = deepcopy(score)
optimal_pvi = deepcopy(pvi)

candidates = torch.vstack(candidates).to(torch.double)
scores = torch.vstack(scores).to(torch.double)
for _ in range(5):
    gp = SingleTaskGP(train_X=candidates, train_Y=scores,outcome_transform=Standardize(m=1))
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)
    logEI = LogExpectedImprovement(model=gp, best_f=gp.train_targets.max().detach())
    beta_scaled, _ = optimize_acqf(
        logEI,
        bounds=bounds,
        q=1,
        num_restarts=30, 
        raw_samples=1024
    )
    beta = 100**beta_scaled-0.99
    candidates = torch.cat([candidates,beta_scaled], dim=0).to(torch.double)
    pvi, score = learn_vgm_pvi(beta.item())
    scores = torch.cat([scores,score.reshape((1,1))], dim=0).to(torch.double)
    if score>optimal_score:
        optimal_score = deepcopy(score)
        optimal_pvi = deepcopy(pvi)
pvi = optimal_pvi

# generate figure
pos = t
x_plot = x.to(torch.float64)
w,mu,sigma = pvi.gmm_parameters(x_plot)
w = w.detach()
mu = mu.detach()
sigma = sigma.detach()

upper_posterior, lower_posterior, mean_posterior = [],[],[]
for i in range(len(x_plot)):
    x_theta = x_plot[i]@posterior_samples.T
    sample = torch.distributions.Poisson(x_theta.exp()).sample().numpy()
    lower_posterior.append(np.quantile(sample,q=0.025))
    upper_posterior.append(np.quantile(sample,q=0.975))
    mean_posterior.append(np.mean(sample))

upper, lower, mean = [],[],[]
for i in range(len(x_plot)):
    sample = []
    anz = np.random.multinomial(100000,w[i].numpy())
    for comp in range(len(w[0])):
        if anz[comp]>0:
            x_theta = x_plot[i]@torch.distributions.MultivariateNormal(mu[comp],sigma[comp]).sample((anz[comp],)).T
            sample.append(torch.distributions.Poisson(x_theta.exp()).sample())
    sample = np.hstack(sample) 
    lower.append(np.quantile(sample,q=0.025))
    upper.append(np.quantile(sample,q=0.975))
    mean.append(np.mean(sample))

x_plot2 = x_plot.clone()
x_plot2[:,[3,4,5]] = 1/3
w,mu,sigma = pvi.gmm_parameters(x_plot2)
upper2, lower2, mean2 = [],[],[]
for i in range(len(x_plot)):
    sample = []
    anz = np.random.multinomial(100000,w[i].numpy())
    for comp in range(len(w[0])):
        if anz[comp]>0:
            x_theta = x_plot2[i]@torch.distributions.MultivariateNormal(mu[comp],sigma[comp]).sample((anz[comp],)).T
            sample.append(torch.distributions.Poisson(x_theta.exp()).sample())
    sample = np.hstack(sample) 
    lower2.append(np.quantile(sample,q=0.025))
    upper2.append(np.quantile(sample,q=0.975))
    mean2.append(np.mean(sample))
upper_posterior2, lower_posterior2, mean_posterior2 = [],[],[]
for i in range(len(x_plot2)):
    x_theta = x_plot2[i]@posterior_samples.T
    sample = torch.distributions.Poisson(x_theta.exp()).sample().numpy()
    lower_posterior2.append(np.quantile(sample,q=0.025))
    upper_posterior2.append(np.quantile(sample,q=0.975))
    mean_posterior2.append(np.mean(sample))

color_posterior = "tab:blue"
color_csgmpd = "tab:red"
fig_width = 8
size_labels = 8
size_titles = 10
fig, axs = plt.subplots(2,dpi=800,sharex=True,figsize=(fig_width,((5**.5 - 1) / 2)*fig_width))
axs[0].plot(pos,mean,color=color_csgmpd,label='VGM-PVI',alpha=0.8)
axs[0].fill_between(pos,lower,upper,color=color_csgmpd,alpha=0.1)
axs[0].plot(pos,lower,color=color_csgmpd,linestyle='--',alpha=0.8)
axs[0].plot(pos,upper,color=color_csgmpd,linestyle='--',alpha=0.8)
axs[0].plot(pos,mean_posterior,color=color_posterior,label='posterior predictive',alpha=0.8)
axs[0].fill_between(pos,lower_posterior,upper_posterior,color=color_posterior,alpha=0.1)
axs[0].plot(pos,lower_posterior,color=color_posterior,alpha=0.8,linestyle='--')
axs[0].plot(pos,upper_posterior,color=color_posterior,alpha=0.8,linestyle='--')
axs[0].set_xticks(t[::4])
axs[0].set_xticklabels(['1983','1984','1985','1986','1987','1988','1989','1990'])
axs[0].grid(alpha=0.3)
axs[0].scatter(pos,y,s=4,alpha=1,color='black',label='data')
axs[0].legend(fontsize='small')
axs[0].set_ylabel('number of cases')
axs[0].text(-0.05, 1.05,'A)', transform=axs[0].transAxes, weight='bold')
axs[1].plot(pos,mean2,color=color_csgmpd,label='CGMPD',alpha=0.8)
axs[1].fill_between(pos,lower2,upper2,color=color_csgmpd,alpha=0.1)
axs[1].plot(pos,lower2,color=color_csgmpd,linestyle='--',alpha=0.8)
axs[1].plot(pos,upper2,color=color_csgmpd,linestyle='--',alpha=0.8)
axs[1].plot(pos,mean_posterior2,color=color_posterior,label='posterior predictive',alpha=0.8)
axs[1].fill_between(pos,lower_posterior2,upper_posterior2,color=color_posterior,alpha=0.1)
axs[1].plot(pos,lower_posterior2,color=color_posterior,alpha=0.8,linestyle='--')
axs[1].plot(pos,upper_posterior2,color=color_posterior,alpha=0.8,linestyle='--')
axs[1].grid(alpha=0.3)
axs[1].set_ylabel('number of cases')
axs[1].text(-0.05, 1.05,'B)', transform=axs[1].transAxes, weight='bold')
plt.show()