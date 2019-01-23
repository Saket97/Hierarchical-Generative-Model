import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--num',type=int,default=10)
FLAGS, unparsed = parser.parse_known_args()


def _get_samples_mvn(mu, sigma, n=1):
    """ 
    Returns n samples from multivariate normal distribution

    Return: A matix of shape (n x shape of mu)
    """
    L = np.linalg.cholesky(sigma)
    t = np.random.normal(size=(n,mu.shape[0]))
    t = np.matmul(L,t.T).T+mu
    return t

def get_samples_mvn(mu, sigma, n=1):
    """
        Get n samples from each of the MVNs defined by the rows of mu with covariance sigma
    """
    samples = []
    for i in range(mu.shape[0]):
        t = _get_samples_mvn(mu[i], sigma, n)
        samples.append(t)

    return np.stack(samples, axis=0)

def get_input(A,X,Psi):
    
    Sigma_inv = np.matmul(A.T, (1.0/np.expand_dims(Psi+1e-5,axis=1))*A)
    Sigma = np.linalg.inv(Sigma_inv)
    
    mu = np.matmul(Sigma.T, np.matmul(A.T, (1.0/np.expand_dims(Psi+1e-5,axis=1))*X.T )  ) # (nfactor x nfactor) (nfactor,D) (D x N) = (nfactor x N)
    mu = mu.T # (N x nfactor)

    return mu, Sigma

def get_log_likelihood(X_test, A, Psi):
    mu, Sigma = get_input(A,X_test,Psi)
    z_samples = get_samples_mvn(mu, Sigma, n=100) # (32,100,60)
    means = np.matmul(z_samples, A.T) #(32,100,5000)
    means = np.transpose(means, axes=[1,0,2]) #(100,32,5000)
    prec = 1/(Psi+1e-5)
    t = (X_test-means)
    t1 = t*prec*t
    t1 = -0.5*np.sum(t1, axis=2) #(100,32)
    t2 = -0.5*X_test.shape[1]*np.sum(np.log(Psi+1e-5))
    # t2 = 0.5*np.expand_dims(np.sum(np.log(1e-5+prec)), axis=1)
    t3 = -X_test.shape[1]*0.5*np.log(2*np.pi)
    x_post_prob_log_test = t1+t2+t3
    return np.mean(np.mean(x_post_prob_log_test,axis=0),axis=0)

    
raw_data = np.load("/opt/data/saket/gene_data/data/mod_gauss_data.npy")
m = np.mean(raw_data, axis=0)
raw_data = (raw_data-m)
A = np.loadtxt("mA.txt")
A = A[:,1:]
Psi = np.loadtxt("mPsi.txt")

lik_list = []
for i in range(1,11):
    X_test = raw_data[(i-1)*32:i*32]
    lik_list.append(get_log_likelihood(X_test,A,Psi))
np.save("test_lik.npy",np.array(lik_list))
# X_test = raw_data[9*32:10*32]

# mu_test, Sigma = get_input(A,X_test,Psi)
# z_samples = get_samples_mvn(mu_test, Sigma, n=100) # (32,100,60)
# means_list = []
# for i in range(100):
#     means_list.append(np.matmul(np.random.normal(size=(32,60)),A.T))
# lik_list = []
# for means in means_list:
#     t1 = -16*np.log(2*np.pi) - 16*60*np.sum(np.log(Psi)) - 0.5*np.sum((1/(Psi+1e-5))*(X_test-means)*(X_test-means))
#     lik_list.append(t1)
# print(np.mean(lik_list))
