import numpy as np

"""I this file I have created the models of HMM and GMM """
class GMM:
    def __init__(self, n_components=5, max_iter=50, tol=1e-3):
        self.K = n_components
        self.max_iter = max_iter
        self.tol = tol
        
    def _logsumexp(self, a, axis=None, keepdim=False):
        a_max = np.max(a, axis=axis, keepdim=True)
        out = np.log(np.sum(np.exp(a -a_max),axis=axis,keepdim=keepdim))
        if not keepdim:
            a_max = np.squeeze(a_max, axis=axis)
            
        return a_max + out

    def _log_gaussian(self, X, mean, cov):
        n_features = X.shape[1]
        diff = X - mean

        inv_cov = np.linalg.inv(cov)
        sign, log_det = np.linalg.slogdet(cov)
        mahalanobis = np.sum(np.dot(diff, inv_cov) * diff, axis=1)
        
        log_prob = -0.5 * (n_features * np.log(2 * np.pi) + log_det + mahalanobis)
        return log_prob

    def fit(self, X):
        n_samples, n_features = X.shape
        self.means = X[np.random.choice(n_samples, self.K, replace=False)]
        self.covs = np.zeros((self.K, n_features, n_features)) 
        self.weights = np.zeros(self.K)
        # K means for better accuracy instead of random initilisation 
        for _ in range(10):
            distances = np.zeros((n_samples, self.K))
            for k in range(self.K):
                distances[:, k] = np.sum((X - self.means[k])**2, axis=1)
            
            labels = np.argmin(distances, axis=1)
            
            for k in range(self.K):
                cluster_data = X[labels == k]
                if len(cluster_data) > 0:
                    self.means[k] = np.mean(cluster_data, axis=0)
        
        for k in range(self.K):
            cluster_data = X[labels == k]
            if len(cluster_data) > 1:
                self.covs[k] = np.cov(cluster_data, rowvar=False) 
                self.covs[k] += np.eye(n_features) * 1e-4
                self.weights[k] = len(cluster_data) / n_samples
            else:
                self.covs[k] = np.cov(X, rowvar=False) + np.eye(n_features) * 1e-4
                self.weights[k] = 1e-6
                
        self.weights /= np.sum(self.weights)
        self.log_weights = np.log(self.weights)
        log_likelihood_old = -np.inf
        
        for iteration in range(self.max_iter):
            log_resp = np.zeros((n_samples, self.K))
            for k in range(self.K):
                log_resp[:, k] = self.log_weights[k] + self._log_gaussian(X, self.means[k], self.covs[k])
            
            log_prob_norm = self._logsumexp(log_resp, axis=1, keepdim=True)
            log_resp -= log_prob_norm
            resp = np.exp(log_resp) 
            
            N_k = np.sum(resp, axis=0)
            for k in range(self.K):
                self.means[k] = np.sum(resp[:, k:k+1] * X, axis=0) / N_k[k]
                
                diff = X - self.means[k]
                self.covs[k] = np.dot((resp[:, k:k+1] * diff).T, diff) / N_k[k]
                self.covs[k] += np.eye(n_features) * 1e-4
                self.weights[k] = N_k[k] / n_samples
                
            self.log_weights = np.log(self.weights)
            
            log_likelihood_new = np.mean(log_prob_norm)
            if np.abs(log_likelihood_new - log_likelihood_old) < self.tol:
                break
            log_likelihood_old = log_likelihood_new

    def score(self, X):
        log_resp = np.zeros((X.shape[0], self.K))
        for k in range(self.K):
            log_resp[:, k] = self.log_weights[k] + self._log_gaussian(X, self.means[k], self.covs[k])
        
        return np.sum(self._logsumexp(log_resp, axis=1))


class VectorQuantizer:
    def __init__(self, n_clusters=32, max_iters=100, tol=1e-4):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tol = tol
        self.centroids = None
        
    def fit(self, features):
        n_samples, n_features = features.shape
        np.random.seed(42) 
        random_indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.centroids = features[random_indices].copy()
        
        for iteration in range(self.max_iters):
            distances = np.linalg.norm(features[:, np.newaxis] - self.centroids, axis=2)
            
            labels = np.argmin(distances, axis=1)
            new_centroids = np.zeros_like(self.centroids)
            for k in range(self.n_clusters):
                cluster_points = features[labels == k]
                if len(cluster_points) > 0:
                    new_centroids[k] = np.mean(cluster_points, axis=0)
                else:
                    new_centroids[k] = features[np.random.choice(n_samples)]
                    
            shift = np.linalg.norm(new_centroids - self.centroids)
            self.centroids = new_centroids
            
            if shift < self.tol:
                break
                
    def transform(self, features):
        distances = np.linalg.norm(features[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)

class HMM:
    def __init__(self, n_states=5, n_symbols=128):
        self.n_states = n_states
        self.n_symbols = n_symbols
        self.pi = np.zeros(n_states)
        self.pi[0] = 1.0 
        
        self.A = np.zeros((n_states, n_states))
        for i in range(n_states):
            if i == n_states - 1:
                self.A[i, i] = 1.0 
            else:
                self.A[i, i] = 0.5    
                self.A[i, i+1] = 0.5   
                
        np.random.seed(42)
        self.B = np.random.rand(n_states, n_symbols)
        self.B = self.B / np.sum(self.B, axis=1, keepdim=True)

    def fit(self, sequences, epochs=10):
        for _ in range(epochs):
            pi_num = np.zeros(self.n_states)
            A_num = np.zeros((self.n_states, self.n_states))
            A_den = np.zeros(self.n_states)
            B_num = np.zeros((self.n_states, self.n_symbols))
            B_den = np.zeros(self.n_states)
            
            for seq in sequences:
                T = len(seq)
                if T == 0: 
                    continue
                alpha = np.zeros((T, self.n_states))
                c = np.zeros(T) 
                
                alpha[0] = self.pi * self.B[:, seq[0]]
                c[0] = np.sum(alpha[0]) + 1e-10
                alpha[0] /= c[0]
                
                for t in range(1, T):
                    alpha[t] = np.dot(alpha[t-1], self.A) * self.B[:, seq[t]]
                    c[t] = np.sum(alpha[t]) + 1e-10
                    alpha[t] /= c[t]
                    
                beta = np.zeros((T, self.n_states))
                beta[-1] = 1.0
                
                for t in range(T-2, -1, -1):
                    beta[t] = np.dot(self.A, self.B[:, seq[t+1]] * beta[t+1])
                    beta[t] /= c[t+1]
                    
                gamma = alpha * beta
                gamma = gamma / (np.sum(gamma, axis=1, keepdim=True) + 1e-10)
                
                xi = np.zeros((T-1, self.n_states, self.n_states))
                for t in range(T-1):
                    xi[t] = (alpha[t].reshape(-1, 1) * self.A * self.B[:, seq[t+1]] * beta[t+1])
                    xi[t] /= (np.sum(xi[t]) + 1e-10)
                    
                pi_num += gamma[0]
                A_num += np.sum(xi, axis=0)
                A_den += np.sum(gamma[:-1], axis=0)
                
                for t in range(T):
                    B_num[:, seq[t]] += gamma[t]
                B_den += np.sum(gamma, axis=0)
                
            self.pi = pi_num / (np.sum(pi_num) + 1e-10)
            self.A = A_num / (A_den.reshape(-1, 1) + 1e-10)
            self.A = np.triu(self.A) 
            self.A = self.A / (np.sum(self.A, axis=1, keepdim=True) + 1e-10)
            
            smoothing_factor = 1e-6
            self.B = (B_num + smoothing_factor) / (B_den.reshape(-1, 1) + (smoothing_factor * self.n_symbols))

    def score(self, seq):
        T = len(seq)
        if T == 0: 
            return -np.inf
    
        alpha = self.pi * self.B[:, seq[0]]
        c = np.sum(alpha) + 1e-10
        alpha /= c
        log_prob = np.log(c)
        for t in range(1, T):
            alpha = np.dot(alpha, self.A) * self.B[:, seq[t]]
            c = np.sum(alpha) + 1e-10
            alpha /= c
            log_prob += np.log(c)
        return log_prob