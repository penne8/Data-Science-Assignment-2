import numpy as np

from Kmeans import Kmeans
# normalize the data from [0-255] to [0-1]
from loadMNIST_py import show_images


# B)


def normalization(vectors):
    normalized_vectors = []
    for i in range(len(vectors)):
        normalized_vectors.append(np.array(vectors[i]).flatten())
        for j in range(len(vectors[i])):
            normalized_vectors[i][j] /= 255
    return normalized_vectors


# C)
# find and return a cluster identifier
def cluster_identifier(initial_centroids, trainImages, trainLabels):
    trained_data_clustered, centroids = Kmeans(initial_centroids, normalization(trainImages)).run()
    data_size = trained_data_clustered.shape[0]
    k = initial_centroids.shape[0]
    popular = np.zeros((k, k))
    for i in range(data_size):
        popular[trained_data_clustered[i]][trainLabels[i]] += 1
    return centroids, np.argmax(popular, axis=1)


# D)
# classify each Image to its closest centroid
def classify(centroids, testImages):
    test_data = np.array(normalization(testImages))
    data_size = test_data.shape[0]
    k = centroids.shape[0]
    distances = np.zeros((data_size, k))
    for i in range(k):
        # find the distance of each vector to each centroid
        distances[:, i] = np.linalg.norm(test_data - centroids[i], axis=1)
    # find the index of the closest centroid cluster
    return np.argmin(distances, axis=1)


def run(initial_centroids, trainImages, trainLabels, testImages, testLabels):
    k = initial_centroids.shape[0]

    # B+C)
    centroids, identifier = cluster_identifier(initial_centroids, trainImages, trainLabels)
    # printing the centroids as images:
    centroids_show = []
    centroids_true_labels = []
    for i in range(0, 10):
        centroids_show.append(centroids[i].reshape(28, 28))
        centroids_true_labels.append('centroid [' + str(i) + '] = represent cluster: ' + str(identifier[i]))
    show_images(centroids_show, centroids_true_labels)
    print("the cluster identifier table for the current run: ")
    print(identifier)

    # D)
    classified = classify(centroids, testImages)

    # D+E+F)
    # an initialized centroid for the next run:
    new_initial_centroids = np.zeros(initial_centroids.shape)

    # find our true estimations
    test_data_size = len(testImages)
    count = 0
    for i in range(test_data_size):
        if identifier[classified[i]] == testLabels[i]:
            new_initial_centroids[testLabels[i]] = np.array(testImages[i]).flatten()
            count += 1
    print("our percentage of true estimations:", count / test_data_size * 100, "%")
    return new_initial_centroids
