import torch
import numpy as np
import matplotlib.pyplot as plt
from vgm_pvi import VGM_PVI
from copy import copy
import pymc as pm


# generate data
n = 1000
n_test = 100000

x = 4*torch.rand((n,2))-2
y = 1.*(torch.rand((n,1))>.5)
y[(x[:,0]<0)*(x[:,1]>0)] = 0
y[(x[:,0]>0)*(x[:,1]<0)] = 1

x_test = 4*torch.rand((n_test,2))-2
y_test = 1.*(torch.rand((n_test,1))>.5)
y_test[(x_test[:,0]<0)*(x_test[:,1]>0)] = 0
y_test[(x_test[:,0]>0)*(x_test[:,1]<0)] = 1
   
# train PVI for different choices of bet 
betas = [0.01,0.1,0.25,0.5,1.,10,100]
fitted_pvis = []
llpds = []
for beta in betas:
    pvi = VGM_PVI(p=len(x[0]),k_init=10,model='Bernoulli',prior='Gaussian',additional_parameters={'tau2': 2.5})
    pvi.train(y=y,x=x,beta=beta)
    fitted_pvis.append(copy(pvi))
    pvi.gmm_parameters(x_test,return_parameters=False)
    llpds.append(1/n_test*pvi.predictive_score(y_test,x_test).item())
    print(llpds[-1])
   
# sample from true posterior
x_np = x.numpy()
y_np = y.flatten().numpy()
with pm.Model() as bayesian_logistic_model:
    theta = pm.Normal('theta', mu=0, sigma=2.5, shape=x_np.shape[1])
    linear_predictor = pm.math.dot(x_np,theta)
    p_success = pm.Deterministic('p_success', pm.math.sigmoid(linear_predictor))
    likelihood = pm.Bernoulli('likelihood', p=p_success, observed=y_np)
    trace = pm.sample(5000, tune=1000, cores=1, chains=4) 
posterior_samples = np.concatenate(trace.posterior["theta"].values,axis=0)


# generate figure
fig_width = 8
size_labels = 8
size_titles = 10
pos = np.linspace(-2,2,500)
xx,yy = np.meshgrid(pos,pos)
x_plot = torch.tensor(np.stack([xx.flatten(),yy.flatten()]).T)
fig, axs = plt.subplot_mosaic([['A','B','C','D'],['E','E','E','E']],dpi=800,figsize=(fig_width,((5**.5 - 1) / 2)*fig_width))#, 
for j in range(3):
    idx = [0,3,-1][j]
    pvi = fitted_pvis[idx]
    w,mu,sigma = pvi.gmm_parameters(x_plot)
    pvi_mean = 0
    for comp in range(pvi.k):
        x_theta = (x_plot@pvi.mu[comp]).unsqueeze(0)+pvi.quadrature_nodes.unsqueeze(1)*(x_plot*(x_plot@pvi.sigma[comp])).sum(axis=1).sqrt().unsqueeze(0)
        inner = pvi.quadrature_weights.unsqueeze(1)*torch.distributions.Bernoulli(logits=x_theta).mean
        pvi_mean += pvi.w[:,comp]*inner.sum(axis=0)
    pvi_mean = pvi_mean.numpy().reshape(xx.shape)
    ax = axs[['A','B','C'][j]]
    ax.contourf(xx,yy,pvi_mean,50,cmap='Blues',vmin=0,vmax=1)
    for filled, label in [(False, 0), (True, 1)]:
        m = (y == label).flatten()
        ax.scatter(x[m, 0], x[m, 1], marker='o' if filled else '^',facecolors='none',edgecolors='black',alpha=0.6,s=6)
    ax.grid(alpha=0.3)
    ax.set_title(r'$\beta='+str(betas[idx])+'$',size=size_titles)
    ax.set_aspect('equal')
    ax.text(-0.05, 1.05,['A)','B)','C)'][j], transform=ax.transAxes, weight='bold')
x_plot@posterior_samples.mean(axis=0)
cs = axs['D'].contourf(xx,yy,torch.special.expit(x_plot@posterior_samples.mean(axis=0)).reshape(xx.shape),50,cmap='Blues',vmin=0,vmax=1)
for filled, label in [(False, 0), (True, 1)]:
    m = (y == label).flatten()
    axs['D'].scatter(x[m, 0], x[m, 1], marker='o' if filled else '^',facecolors='none',edgecolors='black',alpha=0.6,s=6)
axs['D'].grid(alpha=0.3)
axs['D'].set_title('posterior',size=size_titles)
axs['D'].set_aspect('equal')
axs['D'].text(-0.05, 1.05,'D)', transform=axs['D'].transAxes, weight='bold')
for j in range(4):
    ax = axs[['A','B','C','D'][j]]
    ax.set_xticks([-2,0,2],labels=[-2,0,2],fontsize=size_labels)
    ax.set_yticks([-2,0,2],labels=[-2,0,2],fontsize=size_labels)
axs['E'].plot(betas,llpds,color='black')
axs['E'].scatter(betas,llpds,color='black')
axs['E'].grid(alpha=0.3)
axs['E'].set_xscale('symlog')
axs['E'].text(-0.05, 1.05,'E)', transform=axs['E'].transAxes, weight='bold')
axs['E'].set_ylabel('llpd')
axs['E'].set_xlabel(r'$\beta$')
axs['E'].tick_params(axis='both', which='major', labelsize=size_labels)
for j in range(3):
    idx = [0,3,-1][j]
    axs['E'].annotate(['A)','B)','C)'][j],xy=(betas[idx],llpds[idx]),xycoords='data',xytext=([5,5,1][j],[-3,1,5][j]),textcoords='offset points')
plt.subplots_adjust(wspace=0.25, hspace=0.0) 
cb = fig.colorbar(cs, ax=[axs['A'],axs['B'],axs['C'],axs['D']],shrink=0.6,location='right',pad=0.02)
cb.set_ticks([0,.2,.4,.6,.8,1.],labels=[0,.2,.4,.6,.8,1.],fontsize=size_labels)
cb.set_label('Probability')
plt.show()
