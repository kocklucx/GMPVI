import torch
import numpy as np
import math

class VGM_PVI:
    """
    Inputs:
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
    """
    def __init__(self,p,k_init,model,prior,additional_parameters=None,diagonal_covariances=False):
        super().__init__()
        self.p = p
        self.k = k_init
        self.model = model
        self.dtype=torch.float64
        
        self.eta = torch.cat([torch.zeros(1,self.p,dtype=self.dtype),torch.randn((self.k-1,self.p),dtype=self.dtype)])
        
        if model=='Gaussian_var':
            self.p += 1
        self.mu = torch.randn((self.k,self.p),dtype=self.dtype)
        self.l_ast = torch.randn((self.k,self.p,self.p),dtype=self.dtype)
        self.l_ast[:,torch.triu_indices(row=self.p, col=self.p, offset=1)[0],torch.triu_indices(row=self.p, col=self.p, offset=1)[1]] = 0
        
        self.prior = prior
        if prior=='Gaussian':
            self.tau2 = torch.tensor(additional_parameters['tau2'],dtype=self.dtype)
        if prior=='multGaussian':
            self.omega_inv = torch.linalg.inv(additional_parameters['Omega']).to(dtype=self.dtype)
        if model=='Gaussian':
            self.gaussian_variance = torch.tensor(additional_parameters['sigma2'],dtype=self.dtype)
        quadrature_nodes, quadrature_weights = np.polynomial.hermite.hermgauss(10) 
        self.quadrature_nodes = torch.tensor(quadrature_nodes, dtype=self.dtype)*torch.tensor(2).sqrt()
        self.quadrature_weights = torch.tensor(quadrature_weights, dtype=self.dtype)/torch.tensor(torch.pi).sqrt()
        
        self.diagonal_covariances = diagonal_covariances
        if diagonal_covariances==True:
            self.l_ast[:,torch.tril_indices(row=self.p, col=self.p, offset=-1)[0],torch.tril_indices(row=self.p, col=self.p, offset=-1)[1]] = 0
            
    def gmm_parameters(self,x,return_parameters=True):
        x = x.to(dtype=self.dtype)
        w = x@self.eta.T
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
        """
        Lower bound for the entropy of a GMM
        """
        entropy = 0
        for i in range(self.k):
            inner = 0
            for j in range(self.k):
                zij = torch.distributions.MultivariateNormal(self.mu[j],self.sigma[i]+self.sigma[j]).log_prob(self.mu[i]).exp()
                inner += self.w_bar[j]*zij
            entropy += self.w_bar[i] * inner.log()
        return -entropy  
    
    def predictive_score(self,y,x):
        x = x.to(dtype=self.dtype)
        y = y.to(dtype=self.dtype)
        if self.model == 'Gaussian':
            s = 0
            for comp in range(self.k):
                s+=self.w[:,comp]*torch.distributions.Normal(x@self.mu[comp],((x*(x@self.sigma[comp])).sum(axis=1)+self.gaussian_variance).sqrt()).log_prob(y.flatten()).exp()
            return s.clip(1e-5,None).log().sum()
        
        if self.model=='Gaussian_var':
            s = 0
            for comp in range(self.k):
                mu_beta = self.mu[comp][:self.p-1]
                mu_tau = self.mu[comp][-1]
                sigma_betabeta = self.sigma[comp][:-1,:-1]
                sigma_taubeta = self.sigma[comp][-1,:-1]
                sigma_tautau = self.sigma[comp][-1,-1]
                mean = x@(mu_beta+sigma_taubeta/sigma_tautau.sqrt()*self.quadrature_nodes.unsqueeze(1)).T
                cov = ((x*(x@sigma_betabeta)).sum(axis=1)+(mu_tau+sigma_tautau.sqrt()*self.quadrature_nodes.unsqueeze(1)).exp()).T
                s+=self.w_bar[comp]*(self.quadrature_weights*torch.distributions.Normal(mean,cov).log_prob(y).exp()).sum(axis=1)
            return s.clip(1e-5,None).log().sum()
        
        if self.model == 'Poisson':
            s=0
            for comp in range(self.k):
                x_theta = (x@self.mu[comp]).unsqueeze(0)+self.quadrature_nodes.unsqueeze(1)*(x*(x@self.sigma[comp])).sum(axis=1).sqrt().unsqueeze(0)
                inner = self.quadrature_weights.unsqueeze(1)*torch.distributions.Poisson(x_theta.exp()).log_prob(y.flatten()).exp()
                s += self.w[:,comp]*inner.sum(axis=0).clamp_min(1e-300)
            return s.log().sum()
        
        if self.model == 'Bernoulli':
            s=0
            for comp in range(self.k):
                x_theta = (x@self.mu[comp]).unsqueeze(0)+self.quadrature_nodes.unsqueeze(1)*(x*(x@self.sigma[comp])).sum(axis=1).sqrt().unsqueeze(0)
                inner = self.quadrature_weights.unsqueeze(1)*torch.distributions.Bernoulli(logits=x_theta).log_prob(y.flatten()).exp()
                s += self.w[:,comp]*inner.sum(axis=0).clamp_min(1e-300)
            return s.log().sum()

    def expected_log_prior(self):
        if self.prior == 'Gaussian':
            l = 0
            for comp in range(self.k):
                l+=self.w_bar[comp]*(-.5*self.p*torch.log(2*torch.pi*self.tau2)-1/(2*self.tau2)*(torch.diag(self.sigma[comp]).sum()+self.mu[comp].pow(2).sum()))
            return l
        if self.prior == 'multGaussian':
            l = 0
            for comp in range(self.k):
                l+=self.w_bar[comp]*(-.5*self.p*torch.log(2*torch.tensor(torch.pi))+1/2*torch.linalg.det(self.omega_inv).log()-1/2*self.mu[comp].T@self.omega_inv@self.mu[comp]+torch.diag(self.omega_inv@self.sigma[comp]).sum())
            return l
        if self.prior=='Laplace':
            l = 0
            for comp in range(self.k):
                variance = torch.diag(self.sigma[comp])
                mean = self.mu[comp]
                l+=(variance*torch.sqrt(torch.tensor(2/torch.pi))*(-mean.pow(2)/(2*variance)).exp()+mean.abs()*(2*torch.distributions.Normal(0,1).cdf(mean.abs()/variance.sqrt())-1)).sum()
            return l
        
    def expected_log_likelihood(self,y,x):
        x = x.to(dtype=self.dtype)
        y = y.to(dtype=self.dtype)
        if self.model == 'Gaussian':
            n = len(y)
            l = 0
            for comp in range(self.k):
               l+=self.w_bar[comp]*(-.5*n*torch.log(2*torch.pi*self.gaussian_variance)-1/(2*self.gaussian_variance)*((y.flatten()-x@self.mu[comp]).pow(2)+(x*(x@self.sigma[comp])).sum(axis=1)).sum())
            return l
        if self.model=='Gaussian_var':
            n = len(y)
            l = 0
            for comp in range(self.k):
                mu_beta = self.mu[comp][:self.p-1]
                mu_tau = self.mu[comp][-1]
                sigma_betabeta = self.sigma[comp][:-1,:-1]
                sigma_taubeta = self.sigma[comp][-1,:-1]
                sigma_tautau = self.sigma[comp][-1,-1]
                
                l+=self.w_bar[comp]*(-.5*n*math.log(2*math.pi)-n/2*mu_tau-1/2*(-mu_tau+0.5*sigma_tautau).exp()*((y.flatten()-x@(mu_beta-sigma_taubeta)).pow(2)+(x*(x@sigma_betabeta)).sum(axis=1)).sum())              
            return l
        
        if self.model == 'Poisson':
            l = 0
            for comp in range(self.k):
               l+=self.w_bar[comp]*(y*x@self.mu[comp]-(x@self.mu[comp]+.5*(x*(x@self.sigma[comp])).sum(axis=1)).exp()-torch.lgamma(y.flatten()+1)).sum()
            return l
        if self.model == 'Bernoulli':
            l = 0
            for comp in range(self.k):
                x_theta = x@self.mu[comp]+self.quadrature_nodes.unsqueeze(1)*(x*(x@self.sigma[comp])).sum(axis=1)
                l+= self.w_bar[comp]*(self.quadrature_weights.unsqueeze(1)*torch.distributions.Bernoulli(logits=x_theta).log_prob(y.flatten())).sum()
            return l
        
    def sga(self,beta,y,x,n_iter=20000,lr=0.01):
        x = x.to(dtype=self.dtype)
        y = y.to(dtype=self.dtype) 
        
        parameters = [self.eta,self.mu,self.l_ast]
        for par in parameters:
            par.requires_grad = True
        
        mask_eta = torch.ones(self.eta.shape,dtype=self.dtype)
        mask_eta[0] = 0
        self.eta.register_hook(lambda grad: grad * mask_eta)
        mask_l_ast = torch.ones(self.l_ast.shape,dtype=self.dtype)
        mask_l_ast[:,torch.triu_indices(row=self.p, col=self.p, offset=1)[0],torch.triu_indices(row=self.p, col=self.p, offset=1)[1]] = 0
        if self.diagonal_covariances==True:
            mask_l_ast[:,torch.tril_indices(row=self.p, col=self.p, offset=-1)[0],torch.tril_indices(row=self.p, col=self.p, offset=-1)[1]] = 0    
        self.l_ast.register_hook(lambda grad: grad * mask_l_ast)
        
        optimizer = torch.optim.Adam(parameters, lr=lr, maximize=True)  
        for t in range(n_iter):
            optimizer.zero_grad()
            self.gmm_parameters(x,return_parameters=False)
            ps = self.predictive_score(y, x)
            ell = self.expected_log_likelihood(y, x)
            prior = self.expected_log_prior()
            ent = self.entropy_gmm()
            target = ps + beta * (ell + prior + ent)
            target.backward()
            optimizer.step()
        parameters = [self.eta,self.mu,self.l_ast]
        for par in parameters:
            par.requires_grad = False
      
    def split_merge(self,x):
        no_effect = True
        # remove empty components
        self.gmm_parameters(x,return_parameters=False)
        keep = torch.argmax(self.w,axis=1).unique()
        self.eta = self.eta[keep].detach().clone()
        self.mu = self.mu[keep].detach().clone()
        self.l_ast = self.l_ast[keep].detach().clone()
        if self.k != self.mu.shape[0]:
            no_effect = False
        self.k = self.mu.shape[0]
        return no_effect
        
    def train(self,beta,y,x,n_iter=20000,lr=0.01):
        no_effect = False
        while not no_effect:
            self.sga(beta,y,x,n_iter=5000,lr=lr)
            no_effect = self.split_merge(x)
        self.sga(beta,y,x,n_iter=n_iter,lr=lr)
    
    def elpd_waic(self,y,x):
        self.gmm_parameters(x,return_parameters=False)
        x = x.to(dtype=self.dtype)
        y = y.to(dtype=self.dtype)
        if self.model == 'Gaussian':
            elpd = 0
            for i in range(len(x)):
                sample = []
                anz = np.random.multinomial(100000,self.w[i].numpy())
                for comp in range(len(self.w[0])):
                    if anz[comp]>0:
                        x_theta = x[i]@torch.distributions.MultivariateNormal(self.mu[comp],self.sigma[comp]).sample((anz[comp],)).T
                        sample.append(torch.distributions.Normal(x_theta,self.gaussian_variance.sqrt()).log_prob(y[i]))
                sample = torch.hstack(sample) 
                elpd += sample.exp().mean().log()-sample.std().pow(2)   
            
        if self.model == 'Gaussian_var':
            elpd = 0
            for i in range(len(x)):
                sample = []
                anz = np.random.multinomial(100000,self.w[i].numpy())
                for comp in range(len(self.w[0])):
                    if anz[comp]>0:
                        sample_theta = torch.distributions.MultivariateNormal(self.mu[comp],self.sigma[comp]).sample((anz[comp],))
                        sample_beta = sample_theta[:,:-1]
                        sample_sigma = sample_theta[:,-1].exp().sqrt()
                        x_theta = x[i]@sample_beta.T
                        sample.append(torch.distributions.Normal(x_theta,sample_sigma).log_prob(y[i]))
                sample = torch.hstack(sample) 
                elpd += sample.exp().mean().log()-sample.std().pow(2)   
    
        if self.model == 'Poisson':
            elpd = 0
            for i in range(len(x)):
                sample = []
                anz = np.random.multinomial(100000,self.w[i].numpy())
                for comp in range(len(self.w[0])):
                    if anz[comp]>0:
                        x_theta = x[i]@torch.distributions.MultivariateNormal(self.mu[comp],self.sigma[comp]).sample((anz[comp],)).T
                        sample.append(torch.distributions.Poisson(x_theta.exp()).log_prob(y[i]))
                sample = torch.hstack(sample) 
                elpd += sample.exp().mean().log()-sample.std().pow(2)

            
        if self.model == 'Bernoulli':
            elpd = 0
            for i in range(len(x)):
                sample = []
                anz = np.random.multinomial(100000,self.w[i].numpy())
                for comp in range(len(self.w[0])):
                    if anz[comp]>0:
                        x_theta = x[i]@torch.distributions.MultivariateNormal(self.mu[comp],self.sigma[comp]).sample((anz[comp],)).T
                        sample.append(torch.distributions.Bernoulli(logits=x_theta).log_prob(y[i]))
                sample = torch.hstack(sample) 
                elpd += sample.exp().mean().log()-sample.std().pow(2)   
        return elpd
        