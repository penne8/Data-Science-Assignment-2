import numpy as np

import classifier
from loadMNIST_py import MnistDataloader

mnistDataLoader = MnistDataloader(
    'train-images.idx3-ubyte',
    'train-labels.idx1-ubyte',
    't10k-images.idx3-ubyte',
    't10k-labels.idx1-ubyte')
(trainImages, trainLabels), (t10kImages, t10kLabels) = mnistDataLoader.load_data()

print("1st run: ")
initial_centroids = np.random.randn(10, 28 * 28)
classifier.run(initial_centroids, trainImages, trainLabels, t10kImages, t10kLabels)

print("2nd run: ")
initial_centroids = np.random.randn(10, 28 * 28)
classifier.run(initial_centroids, trainImages, trainLabels, t10kImages, t10kLabels)

print("3rd run: ")
initial_centroids = np.random.randn(10, 28 * 28)
initial_centroids = classifier.run(initial_centroids, trainImages, trainLabels, t10kImages, t10kLabels)

print("4th run with chosen initialized centroids: ")
# F)
classifier.run(initial_centroids, trainImages, trainLabels, t10kImages, t10kLabels)
