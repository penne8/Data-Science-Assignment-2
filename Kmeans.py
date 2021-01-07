import numpy as np


# A)
class Kmeans:

    def __init__(self, initial_centroids, vectors):
        self.k = initial_centroids.shape[0]
        self.vectors = np.array(vectors)
        self.vector_size = self.vectors.shape[1]
        self.num_of_vectors = self.vectors.shape[0]
        self.centroids = initial_centroids
        self.clusters = []
        self.should_terminate = False

    def update_clusters(self):

        distances = np.zeros((self.num_of_vectors, self.k))
        for i in range(self.k):
            # find the distance of each vector to each centroid
            distances[:, i] = np.linalg.norm(self.vectors - self.centroids[i], axis=1)
        # find the index of the closest centroid cluster
        self.clusters = np.argmin(distances, axis=1)

    def update_centroids(self):
        new_centroids = np.zeros(self.centroids.shape)

        for i in range(self.k):
            new_centroids[i] = np.mean(self.vectors[self.clusters == i], axis=0)

        error = np.linalg.norm(new_centroids - self.centroids)
        self.should_terminate = error == 0
        self.centroids = new_centroids

    def run(self):
        count = 0
        # start iterating
        while not self.should_terminate:
            count += 1
            self.update_clusters()
            self.update_centroids()
            print("finished ", count, " Iterations.")

        return self.clusters, self.centroids
