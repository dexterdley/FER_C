import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 40})
import pdb


def mean_gibbs_distribution(x, lam1, alpha=1):
    logits = (-1 - lam1 * x)/alpha

    logits = torch.exp(logits)
    return logits/torch.sum(logits)

def solve_mean_lagrange(x, mu, pi, lam=0, max_iter = 20, tol = 1e-15):
    #Implements the Newton Raphson method:
    i = 0
    batch_sz = pi.shape[0]
    lam = torch.zeros(batch_sz).to('cuda')
    old_lam = torch.zeros(batch_sz).to('cuda')

    def fx_mean(lam1, x, mean):
        #pi = mean_gibbs_distribution(x, lam)
        return (x * pi).sum(1) - mean

    def dxfx_mean(lam1, x, alpha=1):
        #pi = mean_gibbs_distribution(x, lam)
        derivative = (x**2/alpha * pi ).sum(1) - (x/alpha * pi ).sum(1).pow(2)
        return derivative

    while abs( fx_mean(lam, x, mu).mean() ) > tol: #run the helper function and check

        lam = old_lam - fx_mean(lam, x, mu)/dxfx_mean(lam,x)  # Newton-Raphson equation
        #print("Iteration" + str(i) + ": x = " + str(lam.mean()) + ", f(x) = " +  str( fx_mean(lam, x, mu).mean() ) )  

          
        old_lam = lam
        i += 1
        
        if i > max_iter or torch.square(lam - old_lam).mean() == 0: #converged
          break

    #pdb.set_trace()
    return torch.clamp(lam, min=torch.zeros(1).to('cuda'), max=torch.ones(1).to('cuda'))

def solve_var_lagrange(x, var, lam=0, max_iter = 20, tol = 1e-15):
    #Implements the Newton Raphson method:
    i = 0
    old_lam = lam

    def fx_var(lam2, x, var):
        return var/np.exp(-1) - np.dot( x**2 , np.exp(-lam2 * x**2 ) )

    def dxfx_var(lam2, x):
        return np.dot( x**4, np.exp(-lam2 * x**2 ) ) + np.exp(-lam2 * x**2 ).sum()

    while abs( fx_var(lam, x, var) ) > tol: #run the helper function and check
        lam = old_lam - fx_var(lam, x, var)/dxfx_var(lam, x)  # Newton-Raphson equation
        #print("Iteration" + str(i) + ": x = " + str(lam) + ", f(x) = " +  str( fx_var(lam, x, var) ) )  
          
        old_lam = lam
        i += 1
        
        if i > max_iter:
          break

    return torch.tensor(lam)

### Solve for multiple Lagrange multipliers (System of non-linear Equations) using Jacobian ###
def solve_multiple_lagrange(x, mu, var, lam1=0, lam2=0, max_iter = 1000, tol = 1e-15):

    def fx_1(lam1, lam2, x, mu):
        C = x - mu
        return mu/np.exp(-1) - np.dot( x , np.exp(-lam1 * x - lam2 * C**2) )

    def fx_2(lam1, lam2, x, mu, var):
        C = x - mu
        return var/np.exp(-1) - np.dot( C**2 , np.exp(-lam1 * x -lam2 * C**2 ) )

    def dxfx1_lam1(lam1, lam2, x, mu, var):
        C = (x - mu)
        return np.dot( x**2 , np.exp(-lam1 * x - lam2 * C**2)) + np.exp(-lam1 * x - lam2 * C**2).sum()

    def dxfx1_lam2(lam1, lam2, x, mu, var):
        C = (x - mu)
        return np.dot( x*C**2 , np.exp(-lam1 * x - lam2 * C**2)) + np.exp(-lam1 * x - lam2 * C**2).sum()

    def dxfx2_lam2(lam1, lam2, x, mu, var):
        C = (x - mu)
        return np.dot( C**4 , np.exp(-lam1 * x - lam2 * C**2)) + np.exp(-lam1 * x - lam2 * C**2).sum()
    
    def compute_fx(lam1, lam2, x, mu, var):
        b = np.empty([2, 1])
        #print(b, b.shape)
        b[0,0] = fx_1(lam1, lam2, x, mu)
        b[1,0] = fx_2(lam1, lam2, x, mu, var)
        
        return -b #this is negative, important!! easy to miss

    def compute_jacobian(lam1, lam2, x, mu):
        J = np.zeros([2,2])
        J[0,0] = dxfx1_lam1(lam1, lam2, x, mu, var)
        J[0,1] = dxfx1_lam2(lam1, lam2, x, mu, var)
        J[1,0] = dxfx1_lam2(lam1, lam2, x, mu, var)
        J[1,1] = dxfx2_lam2(lam1, lam2, x, mu, var)
        return J
    
    lam = torch.tensor([lam1, lam2]).reshape(2,1)
    i = 0
    old_lam = lam

    while abs( compute_fx(lam1, lam2, x, mu, var).mean() ) > tol: #run the helper function and check

        lam1, lam2 = old_lam
        #Put initial guess
        b = compute_fx(lam1, lam2, x, mu, var)
        J = compute_jacobian(lam1, lam2, x, mu)

        # Newton-Raphson equation
        d = np.linalg.solve(J, b) #Jx = b -> x = J^-1b
        lam = torch.tensor(d) + old_lam
        #print("Iteration" + str(i) + ": x = " + str(lam) + ", f(x) = " +  str( compute_fx(lam1, lam2, x, mu, var) ) )

        old_lam = lam
        i += 1
        
        if i > max_iter:
          break

    return torch.tensor(lam[0]), torch.tensor(lam[1])

