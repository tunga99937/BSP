import numpy as np
from numpy.random import seed
from scipy.special import digamma
import time
from numpy import linalg as LA
import utilities as ul
import torch
import torch.nn as nn
# import numdifftools as nd
# from joblib import Parallel, delayed
# from numba import jit, prange

seed(1)
if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'

#For a vector theta ~ Dir(alpha), computes E[log(theta)] given alpha.
def dirichlet_expectation(alpha):
    if(len(alpha.shape) == 1):
        return digamma(alpha) - digamma(sum(alpha))
    return (digamma(alpha) - digamma(np.sum(alpha, axis=1))[:, np.newaxis])

def softmax_convert_torch(matrix, flag):
    """
    """
    matrix_max = torch.max(matrix, dim=flag)[0].view(-1, 1)
    temp = torch.exp(matrix - matrix_max)
    matrix_softmax = temp / torch.sum(temp, dim=flag).view(-1, 1)
    return matrix_softmax

def softmax_convert(matrix, flag): #matrix: mxn
    if(flag == 1):
    #matrix_max: (m, 1)
        matrix_max = np.max(matrix, axis=flag)[:, np.newaxis]
    else:
        matrix_max = np.max(matrix)

    #temp: (m, n)
    temp = np.exp(matrix - matrix_max)
    
    #matrix_softmax: (m, n)
    if(flag == 1):
        matrix_softmax = temp / np.sum(temp, axis=flag)[:, np.newaxis]
    else:
        matrix_softmax = temp / np.sum(temp, axis=flag)

    return matrix_softmax

#Gradient Function
def gradient_f(x, f):
    assert (x.shape[0] >= x.shape[1]), "the vector should be a column vector"
    x = x.astype(float)
    N = x.shape[0]
    gradient = []
    for i in range(N):
        eps = abs(x[i]) *  np.finfo(np.float32).eps 
        xx0 = 1. * x[i]
        f0 = f(x)
        x[i] = x[i] + eps
        f1 = f(x)
        gradient.append(np.array([f1 - f0]).item()/eps)
        x[i] = xx0
    return np.array(gradient).reshape(x.shape)

#Hessian Matrix
def hessian(x, the_func):
    #x: KxL
    L = x.shape[0]
    K = x.shape[1]
    #KxLxL
    hessian = np.zeros((K,L,L)) 
    #KxL
    gd_0 = gradient_f(x.transpose(), the_func).transpose()
    eps = np.linalg.norm(gd_0) * np.finfo(np.float32).eps
    for i in range(L):
        # for j in range(x.shape[1]):
        xx0 = 1.*x[i]
        x[i] = xx0 + eps
        gd_1 =  gradient_f(x.transpose(), the_func).transpose()
        hessian[:,:,i] = ((gd_1 - gd_0)/eps).reshape((K, L))
        print(hessian.shape)
        x[i] =xx0
    return hessian.reshape((K, L, L))

class Model:
    def  __init__(self, path_prior, num_topic, n_term, batch_size, n_infer, learning_rate, 
        alpha, rate, type_model, weight, iters, start):
        self.prior_knowledge = ul.read_prior(path_prior)
        self.num_topic = num_topic  #K=50
        self.n_term = n_term        #V=2823
        self.batch_size = batch_size
        self.n_infer = n_infer
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.weight = weight
        self.iters = iters
        self.length = len(self.prior_knowledge[0]) # L=201

        self.type = type_model
        self.rate = rate

        self.a_t = np.zeros((self.num_topic, self.length, self.length))
        self.b_t = np.zeros((self.num_topic, self.length))

        #need to be learned
        if(start == 0):
            self.pi = softmax_convert(np.random.rand(self.num_topic, self.length), 1) # KxL
        else:
            self.pi = torch.load("params/pi_t.pt")

        #beta_t_kj = pi_t_kj * eta_j
        #50x2823
        self.beta = softmax_convert(np.dot(self.pi, self.prior_knowledge.transpose()), 1)
        
    def update_pi(self, wordinds, wordcnts, minibatch):
        gamma = np.random.gamma(100., 1./100., (self.batch_size, self.num_topic))
        Elogtheta = dirichlet_expectation(gamma)
        expElogtheta = np.exp(Elogtheta)

        pi_t = np.copy(self.pi)
        if(minibatch <= 5):
            p = 1
        else:
            p = self.rate
        
        #create drop matrix
        matrix_drop = np.random.binomial(1, p, size=(self.num_topic, self.n_term)) 
        
        #matrix_drop[matrix_positive[0], matrix_positive[1]] = 1
        matrix_positive = np.where(matrix_drop == 1)

        #matrix_drop[matrix_negative[0], matrix_negative[1]] = 0
        matrix_negative = np.where(matrix_drop == 0)

        #0: Bernoulli, 1: Standard, 2: Init zero
        if(self.type == 0 or self.type == 2):
            matrix_drop[matrix_positive[0], matrix_positive[1]] = 1
        if(self.type == 1):
            matrix_drop[matrix_positive[0], matrix_positive[1]] = 1.0 / p

        ro_t = matrix_drop

        start = time.time()
        for i in range(5):
            if(self.type == 0 or self.type == 2):
                beta_drop_t = softmax_convert(p * np.dot(pi_t, self.prior_knowledge.transpose()), 1)
            if(self.type == 1):
                beta_drop_t = softmax_convert(np.dot(pi_t, self.prior_knowledge.transpose()), 1)
        
            for d in range(self.batch_size):
                indx = wordinds[d]
                cntx = wordcnts[d]
                # Initialize the variational distribution q(theta|gamma) for each mini-batch
                gammad = np.ones(self.num_topic)*self.alpha+float(np.sum(cntx))/self.num_topic
                expElogthetad = np.exp(dirichlet_expectation(gammad))
                betad = beta_drop_t[:,indx]

                for i in range(self.n_infer):
                    # We update local parameter phi and gamma as in Latent Dirichlet Allocation [Blei et al. (2003).]
                    phid = expElogthetad*betad.transpose() 
                    phid /=np.sum(phid, axis=1)[:,np.newaxis]
                    gammad = self.alpha + np.dot(cntx, phid)
                    expElogthetad = np.exp(dirichlet_expectation(gammad))

                expElogtheta[d] = expElogthetad
                gamma[d]=gammad
        
            sum_phi_i = self.compute_sum_phi_i(wordinds, wordcnts, expElogtheta, beta_drop_t)
            sum_phi_i_torch = torch.tensor(sum_phi_i).to(device)

            (pi_t) = self.train_pi(pi_t, ro_t, self.a_t, self.b_t, sum_phi_i_torch, minibatch)
        # f = self.loss_func(sum_phi_i, minibatch)
        # h_t = hessian(pi_t, f)

        h_t = []
        grad_t = []
        for k in range(self.num_topic):
            # print('K = ', k)
            if(minibatch <= 5):
                h_t_k, grad_t_k = self.compute_ht(pi_t[k], 1, sum_phi_i_torch[k], minibatch)
            else:
                h_t_k, grad_t_k  = self.compute_ht(pi_t[k], ro_t[k], sum_phi_i_torch[k], minibatch)
            h_t.append(h_t_k)
            grad_t.append(grad_t_k)
        h_t = np.array(h_t)
        grad_t = np.array(grad_t)

        # h_t = self.compute_ht(pi_t, sum_phi_i_torch, minibatch)

        pi_t_tens = torch.tensor(pi_t.reshape((self.num_topic, 1, self.length))).to(device)
        h_t_tens = torch.tensor(h_t).to(device)

        self.a_t += 1./2 * h_t
        self.b_t += grad_t - torch.bmm(pi_t_tens, h_t_tens).clone().detach().cpu().numpy().reshape((self.num_topic, self.length))
        #self.b_t += - torch.bmm(pi_t_tens, h_t_tens).clone().detach().cpu().numpy().reshape((self.num_topic, self.length))
        
        end = time.time()
        print('Total time update: %f' %(end - start))
        torch.save(pi_t, "params/pi_t.pt")

        self.pi = pi_t
        if(self.type == 0 or self.type == 2):
            self.beta = softmax_convert(p * np.dot(pi_t, self.prior_knowledge.transpose()), 1)
        if(self.type == 1):
            self.beta = softmax_convert(np.dot(pi_t, self.prior_knowledge.transpose()), 1)

    def compute_sum_phi_i(self, wordinds, wordcnts, expElogtheta, beta_drop_t):
        sum_phi = np.zeros((self.num_topic, 1)) # k-element is sum(phi_dnk, d=1...D, n=1...Nd)
        sum_phi_i = np.zeros((self.num_topic, self.n_term)) # kj-element is sum(I[w_dn=j]*phi_dnk, d=1...D, n=1...Nd)

        for d in range(self.batch_size):
            inds = wordinds[d]
            cnts = wordcnts[d]
            expElogtheta_d = expElogtheta[d]
            beta_d = beta_drop_t[:, inds]

            phi_d = expElogtheta_d * beta_d.transpose() # infer phi_d one more time
            phi_d /= np.sum(phi_d, axis=1)[:, np.newaxis] # shape len(ids)xK

            temp = cnts * phi_d.transpose()
            sum_phi += np.sum(temp, axis = 1)[:, np.newaxis]
            sum_phi_i[:, inds] += phi_d.transpose() * cnts

        return sum_phi_i

    def compute_loss(self, pi_t, ro_t, a_t, b_t, sum_phi_i, minibatch):
        beta_hat = torch.matmul(pi_t, torch.tensor(self.prior_knowledge.transpose()).to(device)) * ro_t
        log_beta_hat = torch.nn.functional.log_softmax(beta_hat, dim=1)

        loss_hat = (sum_phi_i * log_beta_hat).sum()

        loss_approx = (torch.matmul(torch.matmul(pi_t.view(self.num_topic, 1, self.length), a_t), pi_t.view(self.num_topic, self.length, 1)) \
                        + torch.matmul(b_t.view(self.num_topic, 1, self.length), pi_t.view(self.num_topic, self.length, 1))).sum()
        loss = 1./minibatch * (self.weight*loss_approx - loss_hat)

        return loss/self.batch_size

    def loss_func(self, sum_phi_i, minibatch):
        #pi_t_k: 1xL
        f = lambda pi_t: (sum_phi_i * np.log(softmax_convert(np.dot(pi_t.transpose(), self.prior_knowledge.transpose()), 1))).sum()

        return f

    def compute_ht(self, pi_t, ro_t, sum_phi_i, minibatch):
        pi_t = torch.tensor(pi_t, requires_grad=True).to(device)
        ro_t = torch.tensor(ro_t, dtype=torch.double).to(device)
        # for k in range(self.num_topic):
        #     pi_t[k] = torch.tensor(pi_t[k]).to(device).requires_grad_(True)
        f = -(sum_phi_i * torch.nn.functional.log_softmax(torch.matmul(pi_t, torch.tensor(self.prior_knowledge.transpose()).to(device)) * ro_t, dim=0)).sum()
        #1xL
        grad = torch.autograd.grad(f, pi_t, retain_graph=True, create_graph=True)
        Print = torch.tensor([], dtype=torch.float64).to(device)
        for anygrad in grad[0]:
            x = torch.autograd.grad(anygrad, pi_t, retain_graph=True)[0]
            Print = torch.cat((Print, x))
        return Print.view(pi_t.size()[0], -1).clone().detach().cpu().numpy(), grad[0].clone().detach().cpu().numpy()

        # pi_t = torch.tensor(pi_t, requires_grad=True).to(device)
        # # for k in range(self.num_topic):
        # #     pi_t[k] = torch.tensor(pi_t[k]).to(device).requires_grad_(True)
        # f = (sum_phi_i * torch.log(softmax_convert_torch(torch.matmul(pi_t, torch.tensor(self.prior_knowledge.transpose()).to(device)), 1))).sum()
        # #KxL
        # grad = torch.autograd.grad(f, pi_t, retain_graph=True, create_graph=True)
        # Print = torch.tensor([], dtype=torch.float64)
        # # for k in range(self.num_topic):
        # #     temp = torch.tensor([])
        # for anygrad in grad[0]:
        #     x = torch.autograd.grad(anygrad, pi_t, retain_graph=True)[0]
        #     Print = torch.cat((Print, x))
        
        # return Print.view(pi_t.size()[0], -1).clone().detach().cpu().numpy()

    def train_pi(self, pi_t, ro_t, a_t, b_t, sum_phi_i, minibatch):
        pi_t_learnt = torch.tensor(pi_t).to(device).requires_grad_(True)
        a_t = torch.tensor(a_t).to(device)
        b_t = torch.tensor(b_t).to(device)
        ro_t = torch.tensor(ro_t, dtype=torch.double).to(device)

        optimizer = torch.optim.Adagrad([pi_t_learnt], lr=self.learning_rate)
        grad_t = np.random.rand(self.num_topic, self.length)
        for i in range(self.iters):
            optimizer.zero_grad()
            criterion = self.compute_loss(pi_t_learnt, ro_t, a_t, b_t, sum_phi_i, minibatch)
            criterion.backward(retain_graph = True)
            grad_t = pi_t_learnt.grad
            if(i==self.iters-1):
                print("loss_pi: ", criterion)
            optimizer.step()
        
        # while True:
        #     pi_t_old = pi_t_learnt
        #     optimizer.zero_grad()
        #     criterion = self.compute_loss(pi_t_learnt, a_t, b_t, sum_phi_i, minibatch)
        #     criterion.backward(retain_graph = True)
        #     grad_t = pi_t_learnt.grad
        #     optimizer.step()
        #     pi_t_new = pi_t_learnt
        #     if(float(torch.norm(pi_t_new - pi_t_old))/(self.num_topic * self.length) < 1e-10):
        #         print("loss_pi: ", criterion)
        #         break

        # k = 0
        # while True:
        #     k = k + 1
        #     optimizer.zero_grad()
        #     criterion = self.compute_loss(pi_t_learnt, a_t, b_t, sum_phi_i, minibatch)
        #     old_loss = criterion
        #     criterion.backward(retain_graph = True)
        #     grad_t = pi_t_learnt.grad
        #     optimizer.step()
        #     pi_t_new = pi_t_learnt
        #     new_loss = self.compute_loss(pi_t_new, a_t, b_t, sum_phi_i, minibatch)
        #     if(float(new_loss - old_loss) > 0 or abs(float(new_loss - old_loss)) < 0.1):
        #         print("loss_pi: ", new_loss)
        #         print(k)
        #         break

        return pi_t_learnt.clone().detach().cpu().numpy()
