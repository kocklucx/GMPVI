import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd

def predict(x_star,m,log_s):
    """
    Approximate posterior over f* under q(f):
    mean: μ* = K_*x K_xx^{-1} m
    cov:  Σ* = K_** − K_*x K_xx^{-1} K_x* + K_*x K_xx^{-1} S K_xx^{-1} K_x*
    Returns mean and marginal variance (diagonal of cov).
    """

    Kxs = kernel(x,x_star)     
    Kss = kernel(x_star,x_star) 
    A = torch.cholesky_solve(Kxs, Lk)
    A = A 
    mean_star = (A.T @ m) 
    base_cov = Kss - Kxs.T @ A 
    S = torch.diag(log_s.exp()**2)
    B = A 
    term = B.T @ S @ B 
    cov_star = base_cov + term
    var_star = torch.diag(cov_star).contiguous()
    return mean_star, var_star

def kernel(xa,xb,gamma=1.0):
    xa_ = xa 
    xb_ = xb 
    d2 = (xa_[:, None, :] - xb_[None, :, :]).pow(2).sum(dim=2)
    return torch.exp(-0.5* gamma * d2)

# load data
x_obs = np.array([
    390, 391, 393, 394, 396, 397, 399, 400, 402, 403, 405, 406, 408, 409, 411, 412,
    414, 415, 417, 418, 420, 421, 423, 424, 426, 427, 429, 430, 432, 433, 435, 436,
    438, 439, 441, 442, 444, 445, 447, 448, 450, 451, 453, 454, 456, 457, 459, 460,
    462, 463, 465, 466, 468, 469, 471, 472, 474, 475, 477, 478, 480, 481, 483, 484,
    486, 487, 489, 490, 492, 493, 495, 496, 498, 499, 501, 502, 504, 505, 507, 508,
    510, 511, 513, 514, 516, 517, 519, 520, 522, 523, 525, 526, 528, 529, 531, 532,
    534, 535, 537, 538, 540, 541, 543, 544, 546, 547, 549, 550, 552, 553, 555, 556,
    558, 559, 561, 562, 564, 565, 567, 568, 570, 571, 573, 574, 576, 577, 579, 580,
    582, 583, 585, 586, 588, 589, 591, 592, 594, 595, 597, 598, 600, 601, 603, 604,
    606, 607, 609, 610, 612, 613, 615, 616, 618, 619, 621, 622, 624, 625, 627, 628,
    630, 631, 633, 634, 636, 637, 639, 640, 642, 643, 645, 646, 648, 649, 651, 652,
    654, 655, 657, 658, 660, 661, 663, 664, 666, 667, 669, 670, 672, 673, 675, 676,
    678, 679, 681, 682, 684, 685, 687, 688, 690, 691, 693, 694, 696, 697, 699, 700,
    702, 703, 705, 706, 708, 709, 711, 712, 714, 715, 717, 718, 720
],dtype=float)

y_obs = np.array([
    -0.050355730, -0.060097060, -0.041900910, -0.050984700, -0.059913450, -0.028423920, -0.059584210, -0.039888810,
    -0.029395820, -0.039494450, -0.047647490, -0.060380000, -0.031230340, -0.038165840, -0.075622690, -0.050017510,
    -0.045729500, -0.077669660, -0.024606410, -0.071331840, -0.013207460, -0.031626150, -0.032474780, -0.088407970,
    -0.070241660, -0.028772630, -0.036967020, -0.101562500, -0.068313730, -0.031757510, -0.053368190, -0.057263810,
    -0.022955150, -0.014791660, -0.025318840, -0.095389440, -0.081261080, -0.064738010, -0.049400040, -0.024539830,
    -0.004223316, -0.046929080, -0.072642600, -0.063754970, -0.048675710, -0.079021940, -0.055083920, -0.036009150,
    -0.008198360, -0.029916380, -0.059044170, -0.043490760, -0.108366700, -0.071644950, -0.108043100, -0.011447560,
    -0.090664970, -0.074388490, -0.088807160, -0.072401300, -0.039412080, -0.084136980, -0.044777780, -0.148665100,
    -0.080272660, -0.054824790, -0.012024890,  0.019348600, -0.083894270, -0.041574770, -0.061091210, -0.060443250,
    -0.082201860, -0.075303490, -0.044809910,  0.008222156, -0.067588090, -0.032499460, -0.021981460, -0.042326210,
    -0.077852130, -0.078061950,  0.026907170, -0.092260960, -0.091590450, -0.004001756, -0.018189330, -0.025276170,
    -0.058424990, -0.052573050, -0.026062480, -0.118087700, -0.052631100, -0.041351480, -0.009199134, -0.083366440,
    -0.012533340, -0.062903320, -0.060180180, -0.103568600, -0.126116600, -0.038870120, -0.056549840, -0.074448420,
    -0.003788664, -0.092039710, -0.055173560, -0.100433700, -0.169438000, -0.064141840, -0.186734800, -0.090860060,
    -0.059187140, -0.081035100, -0.103477600, -0.065673940, -0.182124500, -0.085712700, -0.121604100, -0.134269200,
    -0.193390800, -0.117863900, -0.287109000, -0.273944700, -0.186719400, -0.189195300, -0.294862500, -0.371198400,
    -0.232675300, -0.271973600, -0.275020400, -0.310527100, -0.399157200, -0.425672200, -0.519027200, -0.424584400,
    -0.398271100, -0.346376700, -0.440465100, -0.416947400, -0.366367800, -0.412190200, -0.525884900, -0.458107900,
    -0.460408100, -0.521879200, -0.566426900, -0.708995600, -0.650540200, -0.571982900, -0.401473600, -0.525136200,
    -0.563555800, -0.590039200, -0.424141300, -0.489361900, -0.542304700, -0.655504900, -0.509341000, -0.483083200,
    -0.600755200, -0.683696500, -0.515332200, -0.485786300, -0.663813500, -0.769263200, -0.479689100, -0.772027200,
    -0.540661400, -0.565100900, -0.456521700, -0.419902300, -0.539716900, -0.923823200, -0.640224000, -0.825267700,
    -0.596618300, -0.546587100, -0.423674600, -0.707649500, -0.455401300, -0.675938200, -0.658868800, -0.719863500,
    -0.568205500, -0.639420400, -0.731023700, -0.611648100, -0.617259600, -0.755963700, -0.766384800, -0.712309300,
    -0.708644300, -0.537711200, -0.724290000, -0.621456400, -0.632412100, -0.949553500, -0.675763100, -0.588709700,
    -0.911435300, -0.432679400, -0.859001700, -0.798765300, -0.693147200, -0.886574000, -0.796826100, -0.502526800,
    -0.471670200, -0.780108800, -0.666843100, -0.578347900, -0.787452200, -0.615695600, -0.896760200, -0.707737900,
    -0.672567000, -0.621841300, -0.865761100, -0.557754000, -0.802668400
],dtype=float)

x_mean, x_std = x_obs.mean(), x_obs.std()
x = ((x_obs - x_mean) / x_std).reshape(-1, 1)  # shape (n,1)
y_mean, y_std = y_obs.mean(), y_obs.std()
y = (y_obs - y_mean) / y_std
x = torch.from_numpy(x).double()
y = torch.from_numpy(y).double()
n = x.shape[0]

# initialize Gaussian process kernel
K = (kernel(x,x)+ 1e-6 * torch.eye(n, dtype=torch.double)).detach()
Lk = torch.linalg.cholesky(K).detach()
Kinv = torch.cholesky_solve(torch.eye(n, dtype=torch.double), Lk).detach()
logdetK = 2.0 * torch.sum(torch.log(torch.diag(Lk) + 1e-12)).detach()

x_w = torch.hstack([torch.ones((len(x),1)),x])

k = 5
fig, axs = plt.subplots(2,2,dpi=800,sharex=True,sharey=True)
for axz in range(4):
    beta = [0.01,0.5,1.0,100.0][axz]
    ax = axs.flatten()[axz]

    log_sigma =  -2.5*torch.ones(k)
    # variational parameters
    m = 2*torch.randn((k,n),dtype=torch.double)
    log_s = torch.full((k,n,), -1.0, dtype=torch.double)
    eta = torch.zeros(k,2,dtype=torch.double)
    
    log_sigma.requires_grad = True
    m.requires_grad = True
    log_s.requires_grad = True
    eta.requires_grad = True
    mask_eta = torch.ones(eta.shape,dtype=torch.double)
    mask_eta[0] = 0
    eta.register_hook(lambda grad: grad * mask_eta)
    
    optimizer = optim.Adam([eta,m,log_s], lr=0.01,maximize=True)
    optimizer_sigma = optim.Adam([log_sigma], lr=0.01,maximize=True)
    num_steps = 100000
    for t in range(num_steps):
        sigma = log_sigma.exp()+1e-10
        
        optimizer.zero_grad()
        optimizer_sigma.zero_grad()
        
        w = x_w@eta.T
        w = (w-torch.logsumexp(w, 1).unsqueeze(1)).exp()
        w_bar = w.mean(axis=0)
        
        expected_log_prior = 0
        for comp in range(k):
            const_term = -n/2*math.log(2.0 * math.pi)
            log_det = -0.5*logdetK
            quad_m = -0.5*torch.dot(m[comp],torch.cholesky_solve(m[comp].unsqueeze(1), Lk).squeeze(1))
            trace_term = -0.5*torch.sum(torch.exp(2*log_s[comp]) * torch.diag(Kinv))
            expected_log_prior += w_bar[comp]*(const_term+log_det+quad_m+trace_term)
        
        expected_log_likelihood = 0
        for comp in range(k):
            expected_log_likelihood += w_bar[comp]*(-n/2*math.log(2.0*math.pi*sigma[comp]**2)-1/(2*sigma[comp]**2)*((y-m[comp]).pow(2).sum()+torch.sum(torch.exp(2*log_s[comp]))))
        
        entropy = 0
        for i in range(k):
            inner = 0
            zi_log = torch.ones(k)
            for j in range(k):
                zij_log = torch.distributions.Normal(m[j],torch.sqrt(torch.exp(2*log_s[i])+torch.exp(2*log_s[j]))+1e-5).log_prob(m[i]).sum()+w_bar[j].log()
                zi_log[j] = zij_log
            entropy += w_bar[i] * torch.logsumexp(zi_log,0)
        entropy *= -1
        
        score = 0
        for comp in range(k):
            score += (w[:,comp]*torch.distributions.Normal(m[comp],torch.sqrt(torch.exp(2*log_s[comp])+sigma[comp]**2)).log_prob(y)).sum()
        
        elbo = expected_log_likelihood+expected_log_prior+entropy
        target = score+beta*elbo  
        
        target.backward()
        optimizer.step()
        if t>= 25000 and t%200==0:
            optimizer_sigma.step()
        
    m.requires_grad = False
    log_s.requires_grad = False
    eta.requires_grad = False
    log_sigma.requires_grad = False
    sigma = log_sigma.exp()+1e-10
    
    with torch.no_grad():
        x_star = torch.linspace(x.min(), x.max(), 1000,dtype=torch.double).reshape(-1, 1)
        x_w_star = torch.hstack([torch.ones((len(x_star),1)),x_star])
        w_star = x_w_star@eta.T
        w_star = (w_star-torch.logsumexp(w_star, 1).unsqueeze(1)).exp()
        
        mean_star,var_star = [],[]
        for comp in range(k):
             m_star, s2_star = predict(x_star,m[comp],log_s[comp])
             mean_star.append(m_star)
             var_star.append(s2_star)
        
        q5,q95,q50,q25,q75,q01,q99 = [],[],[],[],[],[],[]
        for i in range(len(x_star)):
            sample = []
            anz = np.random.multinomial(100000,w_star[i].numpy())
            for comp in range(k):
                if anz[comp]>0:
                    sample.append(np.random.normal(mean_star[comp][i].item(), np.sqrt(var_star[comp][i]+sigma[comp]**2).item(),size=anz[comp]))
            sample = np.hstack(sample)
            q5.append(np.quantile(sample,q=0.05))
            q95.append(np.quantile(sample,q=0.95))
            q50.append(np.quantile(sample,q=0.50))
            q25.append(np.quantile(sample,q=0.25))
            q75.append(np.quantile(sample,q=0.75))
            q01.append(np.quantile(sample,q=0.01))
            q99.append(np.quantile(sample,q=0.99))

        ax.scatter(x,y,s=3,color='black')
        ax.plot(x_star,q5,color='gray',alpha=0.9)
        ax.plot(x_star,q95,color='gray',alpha=0.9)
        ax.plot(x_star,q50,color='gray',alpha=0.9)
        ax.plot(x_star,q25,color='gray',alpha=0.9)
        ax.plot(x_star,q75,color='gray',alpha=0.9)
        ax.plot(x_star,q99,color='gray',alpha=0.9)
        ax.plot(x_star,q01,color='gray',alpha=0.9)
        ax.grid(alpha=0.3)
        ax.set_title(r'$\beta=$'+str(beta))
        ax.text(-0.05, 1.05,['A)','B)','C)','D)'][axz], transform=ax.transAxes, weight='bold')
fig.tight_layout()
plt.show()