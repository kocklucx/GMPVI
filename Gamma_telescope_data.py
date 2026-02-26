from ucimlrepo import fetch_ucirepo 
import pandas as pd
import numpy as np
import pymc as pm
import torch
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from vgm_pvi import VGM_PVI
from copy import deepcopy

# load data + test-train split
magic_gamma_telescope = fetch_ucirepo(id=159)
df = magic_gamma_telescope.data.original
df['y'] = 1*(df['class']=='g')
df = df.drop('class',axis=1)
df['intercept'] = 1
df = df.sample(frac=1,random_state=2025).reset_index(drop=True)
n_train = 12680
df_train = df.iloc[:n_train]
df_test = df.iloc[n_train:]
y_train = np.asarray(df_train['y']).flatten()
y_test = np.asarray(df_test['y']).flatten()
x_train = np.asarray(df_train.drop('y',axis=1))
mean_preprocess = np.mean(x_train[:,:-1],axis=0)
std_preprocess = np.std(x_train[:,:-1],axis=0)
x_train[:,:-1] = (x_train[:,:-1]-mean_preprocess)/std_preprocess
x_test = np.asarray(df_test.drop('y',axis=1))
x_test[:,:-1] = (x_test[:,:-1]-mean_preprocess)/std_preprocess

# true posterior
with pm.Model() as bayesian_logistic_model:
    theta = pm.Normal('theta', mu=0, sigma=2.5, shape=x_train.shape[1])
    linear_predictor = pm.math.dot(x_train,theta)
    likelihood = pm.Bernoulli('likelihood', logit_p=linear_predictor, observed=y_train)
    trace = pm.sample(10000, tune=1000, cores=1, chains=1) #chains=4
posterior_samples = np.concatenate(trace.posterior["theta"].values,axis=0)


y_train = torch.tensor(y_train).unsqueeze(1).to(torch.float64)
x_train = torch.tensor(x_train).to(torch.float64)
betas = [0.01,0.5,1.0,100]
fitted_pvis = []
for j in range(len(betas)):
    beta = betas[j]
    pvi = VGM_PVI(p=len(x_train[0]),k_init=5,model='Bernoulli',prior='Gaussian',additional_parameters={'tau2': 2.5},diagonal_covariances=False)
    pvi.train(y=y_train,x=x_train,beta=beta)
    fitted_pvis.append(deepcopy(pvi))

# print TPR for FPR = 0.01,0.02,0.05,0.1,0.2
x_test = torch.tensor(x_test).to(torch.float64)
roc_pvi = []
for j in range(len(betas)):
    beta = betas[j]
    pvi = fitted_pvis[j]
    w,mu,sigma = pvi.gmm_parameters(x_test)
    pvi_mean = 0
    for comp in range(pvi.k):
        x_theta = (x_test@pvi.mu[comp]).unsqueeze(0)+pvi.quadrature_nodes.unsqueeze(1)*(x_test*(x_test@pvi.sigma[comp])).sum(axis=1).sqrt().unsqueeze(0)
        inner = pvi.quadrature_weights.unsqueeze(1)*torch.distributions.Bernoulli(logits=x_theta).mean
        pvi_mean += pvi.w[:,comp]*inner.sum(axis=0)
    prb_pvi = pvi_mean.numpy()
    fpr_pvi, tpr_pvi, threshold_posterior = metrics.roc_curve(y_test, prb_pvi)
    roc_auc_pvi = metrics.auc(fpr_pvi, tpr_pvi)
    roc_pvi.append((fpr_pvi,tpr_pvi))
    print(beta,np.round(np.interp(np.asarray([0.01,0.02,0.05,0.1,0.2]), fpr_pvi,tpr_pvi),4),len(w[0]))

prb_posterior = torch.special.expit(x_test@posterior_samples.T).mean(axis=1)
fpr_posterior, tpr_posterior, threshold_posterior = metrics.roc_curve(y_test, prb_posterior)
roc_auc_posterior = metrics.auc(fpr_posterior, tpr_posterior)
print('posterior',np.round(np.interp(np.asarray([0.01,0.02,0.05,0.1,0.2]), fpr_posterior,tpr_posterior),4))

# ROC curve
color_obs = 'gray'
color_posterior = "tab:blue"
color_cgmpd = "tab:red"
fig_width = 8
size_labels = 8
size_titles = 10
fig, ax = plt.subplots(1,1,dpi=800,figsize=(fig_width,((5**.5 - 1) / 2)*fig_width))
ax.plot(fpr_posterior, tpr_posterior,alpha=0.7,label='posterior predictive')
for j in range(len(betas)):
    beta = betas[j]
    fpr_pvi, tpr_pvi = roc_pvi[j]
    ax.plot(fpr_pvi, tpr_pvi,alpha=0.7,label=r'VGM-PVI $\beta='+str(beta)+'$')
ax.plot([0,1],[0,1],color='gray',alpha=0.3)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.legend(fontsize='small')
ax.tick_params(axis='both', which='major', labelsize=size_labels)
ax.grid(alpha=0.3)
ax.set_ylabel('True Positive Rate')
ax.set_xlabel('False Positive Rate')
ax.set_aspect('equal', adjustable='box')
fig.tight_layout()
plt.show()

