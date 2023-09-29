import numpy as np 
import pandas as pd
import copy
# IMPORTANT: DO NOT USE ANY OTHER 3RD PARTY PACKAGES
# (math, random, collections, functools, etc. are perfectly fine)


class KMeans:
    def __init__(self, K: int, normalize: bool):
        self.K = K   # clearly two distinct clusters
        self.centroids = None
        self.maxIter = 300
        self.X = None
        self.normalize = normalize
        
                    
    def fit(self, X):
        """
        Estimates parameters for the classifier
        
        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
        """
        if self.normalize:
            X = normalize(X)
        
        self.X = X

        self.centroids = kmeans_plusplus_init(self.X, self.K)

        for _ in range(self.maxIter):
            sortedPoints = [[] for _ in range(self.K)]
            
            for x in self.X.to_numpy():
                # assign data points to closest centroid
                distance = euclidean_distance(x, self.centroids)
                centroidIdx = np.argmin(distance)
                sortedPoints[centroidIdx].append(x)

            prev_centroid = self.centroids
            self.centroids = [np.mean(cluster, axis=0) for cluster in sortedPoints]
            for i, centroid in enumerate(self.centroids):
                if np.isnan(centroid).any():
                    self.centroids[i] = prev_centroid[i]
    
    def predict(self, X):
        """
        Generates predictions
        
        Note: should be called after .fit()
        
        Args:
            X (array<m,n>): a matrix of floats with 
                m rows (#samples) and n columns (#features)
            
        Returns:
            A length m integer array with cluster assignments
            for each point. E.g., if X is a 10xn matrix and 
            there are 3 clusters, then a possible assignment
            could be: array([2, 0, 0, 1, 2, 1, 1, 0, 2, 2])
        """
        centroids = []
        centroidIndexes = []
        for x in X.to_numpy():
            distance = euclidean_distance(x, self.centroids)
            centroidIdx = np.argmin(distance)
            centroids.append(self.centroids[centroidIdx])
            centroidIndexes.append(centroidIdx)
        
        centroidIndexes = np.array(centroidIndexes)

        return centroidIndexes
    
    def get_centroids(self):
        """
        Returns the centroids found by the K-mean algorithm
        
        Example with m centroids in an n-dimensional space:
        >>> model.get_centroids()
        numpy.array([
            [x1_1, x1_2, ..., x1_n],
            [x2_1, x2_2, ..., x2_n],
                    .
                    .
                    .
            [xm_1, xm_2, ..., xm_n]
        ])
        """
        self.centroids = np.array(self.centroids)
        return self.centroids

    
# --- Some utility functions 

def euclidean_distance(x, y):
    """
    Computes euclidean distance between two sets of points 
    
    Note: by passing "y=0.0", it will compute the euclidean norm
    
    Args:
        x, y (array<...,n>): float tensors with pairs of 
            n-dimensional points 
            
    Returns:
        A float array of shape <...> with the pairwise distances
        of each x and y point
    """
    return np.linalg.norm(x - y, ord=2, axis=-1)

def cross_euclidean_distance(x, y=None):
    """
    
    
    """
    y = x if y is None else y 
    assert len(x.shape) >= 2
    assert len(y.shape) >= 2
    return euclidean_distance(x[..., :, None, :], y[..., None, :, :])


def euclidean_distortion(X, z):
    """
    Computes the Euclidean K-means distortion
    
    Args:
        X (array<m,n>): m x n float matrix with datapoints 
        z (array<m>): m-length integer vector of cluster assignments
    
    Returns:
        A scalar float with the raw distortion measure 
    """
    X, z = np.asarray(X), np.asarray(z)
    assert len(X.shape) == 2
    assert len(z.shape) == 1
    assert X.shape[0] == z.shape[0]
    
    distortion = 0.0
    clusters = np.unique(z)
    for _, c in enumerate(clusters):
        Xc = X[z == c]
        mu = Xc.mean(axis=0)
        distortion += ((Xc - mu) ** 2).sum()
        
    return distortion


def euclidean_silhouette(X, z):
    """
    Computes the average Silhouette Coefficient with euclidean distance 
    
    More info:
        - https://www.sciencedirect.com/science/article/pii/0377042787901257
        - https://en.wikipedia.org/wiki/Silhouette_(clustering)
    
    Args:
        X (array<m,n>): m x n float matrix with datapoints 
        z (array<m>): m-length integer vector of cluster assignments
    
    Returns:
        A scalar float with the silhouette score
    """
    X, z = np.asarray(X), np.asarray(z)
    assert len(X.shape) == 2
    assert len(z.shape) == 1
    assert X.shape[0] == z.shape[0]
    
    # Compute average distances from each x to all other clusters
    clusters = np.unique(z)
    D = np.zeros((len(X), len(clusters)))
    for i, ca in enumerate(clusters):
        for j, cb in enumerate(clusters):
            in_cluster_a = z == ca
            in_cluster_b = z == cb
            d = cross_euclidean_distance(X[in_cluster_a], X[in_cluster_b])
            div = d.shape[1] - int(i == j)
            D[in_cluster_a, j] = d.sum(axis=1) / np.clip(div, 1, None)
    
    # Intra distance 
    a = D[np.arange(len(X)), z]
    # Smallest inter distance 
    inf_mask = np.where(z[:, None] == clusters[None], np.inf, 0)
    b = (D + inf_mask).min(axis=1)
    
    return np.mean((b - a) / np.maximum(a, b))

def normalize(d: pd.DataFrame) -> pd.DataFrame:
    '''
    Normalize the data along axis 0
    '''
    mean = np.mean(d, axis=0)
    standard_deviation = np.std(d, axis=0)

    return (d-mean)/standard_deviation

def kmeans_plusplus_init(X: pd.DataFrame, clusters: int) -> np.ndarray:
    '''
    Initialize centroids with KMeans++
    '''
    centroids = np.array([X.iloc[np.random.choice(X.shape[0])]])

    for _ in range(clusters-1):
        distances = np.array([min([euclidean_distance(x, c) for c in centroids]) for x in X.to_numpy()])
        next_centroid = np.array([X.iloc[np.random.choice(X.shape[0], p=distances/distances.sum())]])
        centroids = np.append(centroids, next_centroid, axis=0)

    return centroids