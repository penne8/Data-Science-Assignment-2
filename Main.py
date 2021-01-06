import classifier
from Kmeans import Kmeans
from loadMNIST_py import MnistDataloader, show_images
import numpy as np

mnistDataLoader = MnistDataloader(
    'train-images.idx3-ubyte',
    'train-labels.idx1-ubyte',
    't10k-images.idx3-ubyte',
    't10k-labels.idx1-ubyte')
(trainImages, trainLabels), (t10kImages, t10kLabels) = mnistDataLoader.load_data()

print("1st run: ")
initial_centroids = np.random.randn(10, 28*28)
# classifier.run(initial_centroids, trainImages, trainLabels, t10kImages, t10kLabels)

print("2nd run: ")
# initial_centroids = np.random.randn(10, 28*28)
# classifier.run(initial_centroids, trainImages, trainLabels, t10kImages, t10kLabels)

print("3rd run: ")
# initial_centroids = np.random.randn(10, 28*28)
# initial_centroids = classifier.run(initial_centroids, trainImages, trainLabels, t10kImages, t10kLabels)

print("4th run with chosen initialized centroids: ")
# F)
initial_centroids[0] = np.array(trainImages[12542]).flatten()
initial_centroids[1] = np.array(trainImages[11440]).flatten()
initial_centroids[2] = np.array(trainImages[32606]).flatten()
initial_centroids[3] = np.array(trainImages[37017]).flatten()
initial_centroids[4] = np.array(trainImages[23623]).flatten()
initial_centroids[5] = np.array(trainImages[33749]).flatten()
initial_centroids[6] = np.array(trainImages[45127]).flatten()
initial_centroids[7] = np.array(trainImages[40358]).flatten()
initial_centroids[8] = np.array(trainImages[1236]).flatten()
initial_centroids[9] = np.array(trainImages[29486]).flatten()

# printing the centroids as images:
centroids_show = []
centroids_true_labels = []
for i in range(0, 10):
    centroids_show.append(initial_centroids[i].reshape(28, 28))
    centroids_true_labels.append('centroid [' + str(i) + '] = represent cluster: ' + str(i))
show_images(centroids_show, centroids_true_labels)

classifier.run(initial_centroids, trainImages, trainLabels, t10kImages, t10kLabels)
