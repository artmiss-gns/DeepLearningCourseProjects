import matplotlib.pyplot as plt

def visualize_clusters(X, labels, centroids):
    plt.figure(figsize=(6, 4))

    # Plotting the data points
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50, alpha=0.5)

    # Plotting centroids
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=200)

    plt.title('KMeans Clustering')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.grid(True)
    plt.show()