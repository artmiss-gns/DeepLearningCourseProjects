from typing import List, Literal

import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns 

from sklearn.datasets import make_blobs

from utils import visualize_clusters

class KMeans :
    def __init__(
        self,
        n_clusters:int,
        centroid_init:List[Literal["kmeans++", "random"]] = "random",
        max_iter:int = 10,
    ) -> None:
        self.n_clusters = n_clusters
        self.centroid_init = centroid_init
        self.max_iter = max_iter
    
    @staticmethod
    def get_distance(x1, x2) :
        # Calculating the Distance
        if x1.ndim == 1 and x2.ndim == 1 :
            return np.linalg.norm(x1 - x2)
        else :
            return np.linalg.norm(x1 - x2, axis=1)
    
    def _set_centroids(self) :
        if self.centroids is None : # initializing the centroids for the first time
            if self.centroid_init == "random" :
                chosen_centroids_index = np.random.randint(0, self.n_samples+1, size=self.n_clusters)
                self.centroids = self.X[chosen_centroids_index]
            elif self.centroid_init == "kmeans++" :
                self.centroids = []
                first_centroid = self.X[np.random.randint(0, self.n_samples+1, size=1)].reshape(self.n_features,)
                self.centroids.append(first_centroid)
                for index in range(self.n_clusters-1) :
                    max_distance = 0
                    for sample in self.X :
                        closest_centroid = self.centroids[np.argmin(
                            KMeans.get_distance(
                                sample,
                                np.array(self.centroids).reshape(-1, self.n_features)
                            )
                        )]
                        distance = KMeans.get_distance(
                            sample,
                            closest_centroid
                        )
                        if max_distance < distance :
                            new_centroid_candidate = sample
                            max_distance = distance
                    self.centroids.append(new_centroid_candidate)
                self.centroids = np.array(self.centroids)

        else :
            for index in range(self.n_clusters) :
                new_centroid = self.X[self.clusters == index].mean(axis=0)
                self.centroids[index] = new_centroid

    def fit(self, x) :
        self.X = x
        self.n_samples, self.n_features = x.shape
        self.centroids = None 
        self.clusters = np.empty(shape=(self.n_samples,))

        for _ in range(self.max_iter) :
            self._set_centroids()
            prev_clusters = self.clusters.copy()
            for index, sample in enumerate(self.X) :
                distance = self.get_distance(sample, self.centroids) # distance from eah centroid
                closest_centroid_index = np.argmin(distance)
                self.clusters[index] = closest_centroid_index
            if np.equal(prev_clusters, self.clusters).all() : # check if anything has changed since the last update
                break

    def predict(self, x) :

        return np.argmin([KMeans.get_distance(x, centroid) for centroid in self.centroids], axis=0)

    
    def evaluate(self, X,
        labels,
        method:List[Literal["Silhouette", "Calinski", "wcss"]]
    ) :
        '''
        Silhouette Score :  
            This metric ranges from -1 to 1, the higher the value, the better the clustering\n
            s(i) = (b(i) - a(i)) / max(a(i), b(i)) \n

        Calinski-Harabasz Index  :
            Measures ratio of dispersion between clusters to dispersion within clusters.    
            The higher the value, the better the clustering\n
            score = (trace(B) / (k - 1)) / (trace(W) / (n - k))\n

        WCSS :  
            Within-Cluster Sum of Squares (WCSS)  
            measures the sum of squared distances between each data point and the centroid of its assigned cluster\n
        '''
        if method == "Silhouette":
            silhouette_coeffs = []
            
            for i in range(len(X)):
                cluster_i = X[labels == labels[i]]
                a = np.sum(np.sqrt(np.sum((X[i] - cluster_i)**2, axis=1))) / (len(cluster_i) - 1)
                
                b = np.inf
                for j in range(self.n_clusters):
                    if j != labels[i]:
                        cluster_j = X[labels == j]
                        b = min(b, np.sum(np.sqrt(np.sum((X[i] - cluster_j)**2, axis=1))) / len(cluster_j))
                
                silhouette_coeffs.append((b - a) / max(a, b))
            
            return np.mean(silhouette_coeffs)
        
        elif method == "Calinski":            
            cluster_means = []
            for i in range(self.n_clusters):
                cluster_means.append(np.mean(X[labels == i], axis=0))
            
            overall_mean = np.mean(X, axis=0)
            
            B = 0
            for i in range(self.n_clusters):
                cluster_i = X[labels == i]
                B += len(cluster_i) * np.sum((cluster_means[i] - overall_mean)**2)
            
            W = 0
            for i in range(self.n_samples):
                W += np.sum((X[i] - cluster_means[int(labels[i])])**2)
            
            return (B / (self.n_clusters - 1)) / (W / (self.n_samples - self.n_clusters))
        
        elif method == "wcss":
            cluster_means = np.array([X[labels == label].mean(axis=0) for label in np.arange(self.n_clusters)])
            labels = labels.astype(int)
            wcss = np.sum(np.sum((X - cluster_means[labels])**2, axis=1))
            
            return wcss
        
if __name__ == "__main__":
    X, y = make_blobs(
        n_samples=1000,
        n_features=8,
        centers=5,
        random_state=9090
    )

    model = KMeans(
        n_clusters=5,
        centroid_init="kmeans++",
        # centroid_init="random",
        max_iter=10,
    )
    model.fit(X)
    model.predict(X)


