from Kmeans import Kmeans
from loadMNIST_py import MnistDataloader
import numpy as np

mnistDataLoader = MnistDataloader(
    'train-images.idx3-ubyte',
    'train-labels.idx1-ubyte',
    't10k-images.idx3-ubyte',
    't10k-labels.idx1-ubyte')
(trainImages, trainLabels), (t10kImages, t10kLabels) = mnistDataLoader.load_data()

# B)
numbers = []
k = 10


# normalize the data
for i in range(len(trainImages)):
    numbers.append(np.array(trainImages[i]).flatten())
    for j in range(len(trainImages[i])):
        numbers[i][j] /= 255
kmeans = Kmeans(k, numbers)
trained_data_clustered, centroids = kmeans.run()
data_size = trained_data_clustered.shape[0]

# C)

cluster_identifier = np.zeros(k)
popular = np.zeros((k,k))
for i in range(data_size):
    popular[trained_data_clustered[i]][trainLabels[i]] += 1

cluster_identifier = np.argmax(popular, axis=1)

# D)

numbers = []

# normalize the data
for i in range(len(trainImages)):
    numbers.append(np.array(trainImages[i]).flatten())
    for j in range(len(trainImages[i])):
        numbers[i][j] /= 255
test_data = np.array(numbers)
data_size = test_data.shape[0]

distances = np.zeros((data_size, k))
for i in range(k):
    # find the distance of each vector to each centroid
    distances[:, i] = np.linalg.norm(test_data - centroids[i], axis=1)
# find the index of the closest centroid cluster
test_data_clustered = np.argmin(distances, axis=1)

count = 0
evaluation = np.zeros(data_size)
for i in range(data_size):
    evaluation[i] = cluster_identifier[trained_data_clustered[i]]
    if evaluation[i] == test_data_clustered[i]:
        count += 1

print("our percentage of true estimations: ", count/data_size)

# E)


