import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pymc as pm
import torch
import math
from copy import deepcopy
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import LogExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models.transforms.outcome import Standardize

class VGM_PVI_hierarchical:
    def __init__(self,n,dim_a,dim_b,dim_beta,x_w,x_tilde,x_subject,y,k_init,sigma_a,sigma_b,sigma_eps):
        super().__init__()
        self.n,self.dim_a,self.dim_b,self.dim_beta,self.x_w,self.x_tilde,self.x_subject,self.y = n,dim_a,dim_b,dim_beta,x_w,x_tilde,x_subject,y
        self.sigma_a,self.sigma_b,self.sigma_eps = sigma_a,sigma_b,sigma_eps
        self.k = k_init
        self.p = len(x_tilde[0])
        self.dtype=torch.float64
        self.eta = torch.cat([torch.zeros(1,len(x_w[0]),dtype=self.dtype),torch.randn((self.k-1,len(x_w[0])),dtype=self.dtype)])
        self.mu = torch.randn((self.k,self.p),dtype=self.dtype)
        self.mu[:,0] = y.mean()
        self.l_ast = torch.randn((self.k,self.p,self.p),dtype=self.dtype)
        self.l_ast[:,torch.triu_indices(row=self.p, col=self.p, offset=1)[0],torch.triu_indices(row=self.p, col=self.p, offset=1)[1]] = 0
        self.m = torch.zeros(self.dim_a,dtype=self.dtype)
        self.log_tau = -2*torch.ones(self.dim_a,dtype=self.dtype)
        self.prior_diag = torch.tensor([100**2 for _ in range(dim_beta)]+[sigma_b**2 for _ in range(dim_b)],dtype=self.dtype)
        
    def gmm_parameters(self,return_parameters=True):
        w = self.x_w@self.eta.T
        w = (w-torch.logsumexp(w, 1).unsqueeze(1)).exp()
        L = torch.zeros((self.k, self.p, self.p),dtype=self.dtype)
        idx = torch.arange(self.p)
        L[:, idx, idx] = torch.exp(self.l_ast[:,idx,idx])
        tril_indices = torch.tril_indices(row=self.p, col=self.p, offset=-1)
        L[:, tril_indices[0], tril_indices[1]] = self.l_ast[:,tril_indices[0], tril_indices[1]]
        sigma = torch.matmul(L, L.transpose(-1, -2))+1e-5*torch.eye(self.p)
        self.w = w
        self.sigma = sigma
        self.w_bar = self.w.mean(axis=0)
        if return_parameters:
            return w,self.mu,sigma
    
    def entropy_gmm(self):
        entropy = 0
        for i in range(self.k):
            inner = 0
            for j in range(self.k):
                zij = torch.distributions.MultivariateNormal(self.mu[j],self.sigma[i]+self.sigma[j]).log_prob(self.mu[i]).exp()
                inner += self.w_bar[j]*zij
            entropy += self.w_bar[i] * inner.log()
        return -entropy  
    
    def predictive_score(self):
        s=0
        s = 0
        for comp in range(self.k):
            s+=self.w[:,comp]*torch.distributions.Normal(self.x_tilde@self.mu[comp],((self.x_tilde*(self.x_tilde@self.sigma[comp])).sum(axis=1)+self.sigma_eps**2+self.sigma_a**2).sqrt()).log_prob(y).exp()
        return s.clip(1e-5,None).log().sum()

    def expected_log_prior(self):
        l = -.5*self.p*math.log(2*math.pi)-self.dim_beta*math.log(100)-self.dim_b*math.log(sigma_b)
        for comp in range(self.k):
            l+= self.w_bar[comp]*-.5*(1/self.prior_diag*self.mu[comp].pow(2)).sum()*(1/self.prior_diag*torch.diag(self.sigma[comp])).sum()
        l += -.5*self.n*math.log(2*math.pi*sigma_a**2)-1/(2*sigma_a**2)*(self.m.pow(2).sum()+self.log_tau.exp().pow(2).sum())
        return l

    def expected_log_likelihood(self):
        l = 0
        for comp in range(self.k):
           l+=self.w_bar[comp]*(-.5*n*math.log(2*torch.pi*self.sigma_eps**2)-1/(2*self.sigma_eps**2)*((y-self.x_tilde@self.mu[comp]-self.x_subject@self.m).pow(2)+(self.x_tilde*(self.x_tilde@self.sigma[comp])).sum(axis=1)).sum()*self.dim_b*self.log_tau.exp().pow(2).sum())
        return l

            
    def sga(self,beta,n_iter=20000,lr=0.01):

        parameters = [self.eta,self.mu,self.l_ast,self.m,self.log_tau]
        for par in parameters:
            par.requires_grad = True
        
        mask_eta = torch.ones(self.eta.shape,dtype=self.dtype)
        mask_eta[0] = 0
        self.eta.register_hook(lambda grad: grad * mask_eta)
        mask_l_ast = torch.ones(self.l_ast.shape,dtype=self.dtype)
        mask_l_ast[:,torch.triu_indices(row=self.p, col=self.p, offset=1)[0],torch.triu_indices(row=self.p, col=self.p, offset=1)[1]] = 0
        self.l_ast.register_hook(lambda grad: grad * mask_l_ast)
        
        optimizer = torch.optim.Adam(parameters, lr=lr, maximize=True)  
        for t in range(n_iter):
            optimizer.zero_grad()
            self.gmm_parameters(return_parameters=False)
            ps = self.predictive_score()
            ell = self.expected_log_likelihood()
            prior = self.expected_log_prior()
            ent = self.entropy_gmm()+(math.log(2*math.pi)*self.log_tau).sum()
            target = ps + beta * (ell + prior + ent)
            target.backward()
            optimizer.step()
            #print(target.item())
        parameters = [self.eta,self.mu,self.l_ast,self.m,self.log_tau]
        for par in parameters:
            par.requires_grad = False
      
    def split_merge(self):
        no_effect = True
        # remove empty components
        self.gmm_parameters(return_parameters=False)
        keep = torch.argmax(self.w,axis=1).unique()
        self.eta = self.eta[keep].detach().clone()
        self.mu = self.mu[keep].detach().clone()
        self.l_ast = self.l_ast[keep].detach().clone()
        if self.k != self.mu.shape[0]:
            no_effect = False 
        self.k = self.mu.shape[0]
        return no_effect
        
    def train(self,beta,n_iter=20000,lr=0.01):
        no_effect = False
        while not no_effect:
            self.sga(beta,n_iter=5000,lr=lr)
            no_effect = self.split_merge()
        self.sga(beta,n_iter=n_iter,lr=lr)
    
    def elpd_waic(self):
        self.gmm_parameters()
        elpd = 0
        for i in range(self.n):
            sample = []
            anz = np.random.multinomial(100000,self.w[i].numpy())
            for comp in range(len(self.w[0])):
                if anz[comp]>0:
                    x_theta = self.x_tilde[i]@torch.distributions.MultivariateNormal(self.mu[comp],self.sigma[comp]).sample((anz[comp],)).T
                    x_theta += torch.distributions.Normal((self.x_subject@self.m)[i],(self.x_subject@self.log_tau)[i].exp()).sample((anz[comp],))
                    sample.append(torch.distributions.Normal(x_theta,self.sigma_eps).log_prob(y[i]))
            sample = torch.hstack(sample) 
            elpd += sample.exp().mean().log()-sample.std().pow(2)   
        return elpd

# load and preprocess the data
df = pd.read_csv('temperature_data.csv')
df['Date'] = pd.to_datetime(df['Date'], format="%d/%m/%Y")
df.set_index('Date',inplace=True)
df = df.asfreq('D')
df.rename(columns={'0.5m':'y_1','2m':'y_2','4m':'y_3','6m':'y_4','8m':'y_5','10m':'y_6','12m':'y_7','14m':'y_8','16m':'y_9','18m':'y_10','Bottom':'y_11'},inplace=True)
df['x_0'] = 1
df['x_1'] = np.linspace(0,1,len(df))
df['x_2'] = df['x_1']**2
df['x_3'] = df['x_1']**3
y_cols = [c for c in df.columns if c.startswith('y_')] 
x_cols = [c for c in df.columns if c.startswith('x_')]
df_na = df.dropna().copy()
df_na['indx'] = np.arange(len(df_na))
long_df = df_na.melt(
    id_vars=['indx']+x_cols,
    value_vars=y_cols,
    var_name='y_col',
    value_name='y'
)
subjects = long_df['indx'].astype('category')
long_df['subject_idx'] = subjects.cat.codes
long_df['group'] = long_df['y_col'].str.replace('y_', '', regex=False).astype(int) - 1  # 0..g-1
groups = long_df['group'].astype('category')
long_df['group_idx'] = groups.cat.codes
X = long_df[x_cols].to_numpy()                # shape: (N, p)
y = long_df['y'].to_numpy().astype(float)     # shape: (N,)
subject_idx = long_df['subject_idx'].to_numpy()  # shape: (N,)
group_idx = long_df['group_idx'].to_numpy()      # shape: (N,)
n_subjects = subjects.cat.categories.size
n_groups = groups.cat.categories.size
p = len(x_cols)
N = y.shape[0]

# sample from the conventional posterior
coords = {
    'obs': np.arange(N),
    'feature': x_cols,
    'subject': subjects.cat.categories,
    'group': groups.cat.categories
}


with pm.Model(coords=coords) as model:
    X_data = pm.MutableData('X', X, dims=('obs', 'feature'))
    y_data = pm.MutableData('y', y, dims=('obs',))
    subj_idx = pm.MutableData('subj_idx', subject_idx, dims=('obs',))
    grp_idx = pm.MutableData('grp_idx', group_idx, dims=('obs',))
    beta = pm.Normal('beta', mu=0, sigma=10000, dims=('feature'))  # regression coefficients
    sigma_a = 0.1
    sigma_b = 0.2
    sigma_eps = 0.1
    a = pm.Normal('a', mu=0.0, sigma=sigma_a, dims=('subject'))
    b = pm.Normal('b', mu=0.0, sigma=sigma_b, dims=('group'))  
    mu = pm.math.dot(X_data, beta) + a[subj_idx] + b[grp_idx]
    y_like = pm.Normal('y_like', mu=mu, sigma=sigma_eps, observed=y_data, dims=('obs',))
    trace =  pm.sample(6000, tune=2000, cores=4, chains=4)

sigma_a = 0.1
sigma_b = 0.2
sigma_eps = 0.1
y = torch.tensor(y).to(torch.float64)
n = len(y)
dim_a = len(subjects.unique())
dim_beta = 4
dim_b = 11

x_w = torch.tensor(long_df[x_cols].to_numpy()).to(torch.float64)
x_group = torch.zeros((n,dim_b)).to(torch.float64)
x_group[np.arange(n),group_idx] = 1
x_tilde = torch.cat([x_w,x_group],axis=1)
x_subject = torch.zeros((n,dim_a)).to(torch.float64)
x_subject[np.arange(n),subject_idx] = 1

k_init = 5  
# initalize Bayesian optimization and find the optimal VGM-PVI
bounds = torch.stack([torch.zeros(1), torch.ones(1)]).to(torch.double)
beta_start = 0.4
candidates = [beta_start*torch.ones(1)]

pvi = VGM_PVI_hierarchical(n,dim_a,dim_b,dim_beta,x_w,x_tilde,x_subject,y,k_init,sigma_a,sigma_b,sigma_eps)
pvi.sga(beta=100**beta_start-0.99)
score = pvi.elpd_waic()
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
    pvi = VGM_PVI_hierarchical(n,dim_a,dim_b,dim_beta,x_w,x_tilde,x_subject,y,k_init,sigma_a,sigma_b,sigma_eps)
    pvi.sga(beta=beta.item())
    score = pvi.elpd_waic()
    scores = torch.cat([scores,score.reshape((1,1))], dim=0).to(torch.double)
    if score>optimal_score:
        optimal_score = deepcopy(score)
        optimal_pvi = deepcopy(pvi)
pvi = optimal_pvi
 

# generate figure
sigma_a = 0.1
sigma_b = 0.2
sigma_eps = 0.1
y = torch.tensor(y).to(torch.float64)
n = len(y)
dim_a = len(subjects.unique())
dim_beta = 4
dim_b = 11
    
df['indx'] = np.arange(len(df))
df['Date']=df.index
long_df_plot = df.melt(
    id_vars=['indx','Date']+x_cols,
    value_vars=y_cols,
    var_name='y_col',
    value_name='y',
)

subjects_plot = long_df_plot['indx'].astype('category')
long_df_plot['subject_idx'] = subjects_plot.cat.codes
long_df_plot['group'] = long_df_plot['y_col'].str.replace('y_', '', regex=False).astype(int) - 1  # 0..g-1
groups_plot= long_df_plot['group'].astype('category')
long_df_plot['group_idx'] = groups_plot.cat.codes
X_plot = long_df_plot[x_cols].to_numpy()                # shape: (N, p)    # shape: (N,)
group_idx_plot = long_df_plot['group_idx'].to_numpy()    

x_w_plot = torch.tensor(long_df_plot[x_cols].to_numpy()).to(torch.float64)
x_group_plot = torch.zeros((len(long_df_plot),dim_b)).to(torch.float64)
x_group_plot[np.arange(len(long_df_plot)),group_idx_plot] = 1
x_tilde_plot = torch.cat([x_w_plot,x_group_plot],axis=1)

w,mu,sigma = pvi.gmm_parameters()
w = x_w_plot@pvi.eta.T
w = (w-torch.logsumexp(w, 1).unsqueeze(1)).exp()
plot = (w*(x_tilde_plot@mu.T)).sum(axis=1).detach()

fig_width = 8
fig, axs = plt.subplots(1,3,figsize=(fig_width,.75*((5**.5 - 1) / 2)*fig_width),dpi=300,sharex=True,sharey=True)
for j in range(11):
    axs[0].plot(df['y_'+str(j+1)],color=['dimgray','grey','darkgrey','silver','lightgrey'][j%5])
    axs[1].plot(df.index,plot[group_idx_plot==j],color=['dimgray','grey','darkgrey','silver','lightgrey'][j%5])
    axs[2].plot(df.index,np.asarray(df[x_cols])@np.concatenate(trace.posterior["beta"].values,axis=0).mean(axis=0)+np.concatenate(trace.posterior["b"].values,axis=0).mean(axis=0)[j],color=['dimgray','grey','darkgrey','silver','lightgrey'][j%5])
for j in range(5):
    where_active = list(np.arange(len(df.index))[w[group_idx_plot==0].argmax(axis=1)==j])
    axs[1].scatter(df.index[where_active],27.3*np.ones(len(where_active)),marker='s',s=1,color=['tab:blue','tab:orange','tab:green','tab:red','tab:purple'][j])
    axs[1].scatter(df.index[where_active],27.32*np.ones(len(where_active)),marker='s',s=1,color=['tab:blue','tab:orange','tab:green','tab:red','tab:purple'][j])
    axs[1].scatter(df.index[where_active],27.34*np.ones(len(where_active)),marker='s',s=1,color=['tab:blue','tab:orange','tab:green','tab:red','tab:purple'][j])
    axs[1].scatter(df.index[where_active],27.36*np.ones(len(where_active)),marker='s',s=1,color=['tab:blue','tab:orange','tab:green','tab:red','tab:purple'][j])
for j in range(3):
    axs[j].grid(alpha=0.3)
    axs[j].tick_params(axis='x', labelrotation=45)
    axs[j].text(-0.05, 1.05,['A)','B)','C)'][j], transform=axs[j].transAxes, weight='bold')
axs[0].set_title('Data')
axs[1].set_title('VGM-PVI')
axs[2].set_title('Posterior')
axs[0].set_ylabel('water temperature (C°)')
fig.tight_layout()
plt.show()