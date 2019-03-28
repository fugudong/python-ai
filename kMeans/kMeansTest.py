import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.cluster import KMeans
import numpy as np
from skimage import io

def recreate_image(codebook, labels, w, h):
    d = codebook.shape[1]
    image = np.zeros((w, h, d))
    label_idx = 0
    for i in range(w):
        for j in range(h):
            image[i][j] = codebook[labels[label_idx]]
            label_idx += 1
    return image


if __name__ == '__main__':
    pixel = mpimg.imread('Iris2.jpg')
    width, height, depth = pixel.shape
    print(pixel.shape)
    plt.subplot(2, 1, 1)
    plt.imshow(pixel)

    temp = pixel.reshape((width * height, depth))
    temp = np.array(temp, dtype=np.float64) / 255

    kmeans = KMeans(n_clusters=2,n_jobs=8,random_state=0).fit(temp)
    labels = kmeans.predict(temp)
    kmeans2 = recreate_image(kmeans.cluster_centers_, labels, width, height)

    kmeans = KMeans(n_clusters=3, n_jobs=8, random_state=0).fit(temp)
    labels = kmeans.predict(temp)
    kmeans3 = recreate_image(kmeans.cluster_centers_, labels, width, height)

    kmeans = KMeans(n_clusters=4, n_jobs=8, random_state=0).fit(temp)
    labels = kmeans.predict(temp)
    kmeans4 = recreate_image(kmeans.cluster_centers_, labels, width, height)

    plt.figure(figsize=(15, 10))
    plt.subplot(2, 2, 1)
    plt.axis('off')
    plt.title('Original image')
    plt.imshow(pixel.reshape(width, height, depth))

    plt.subplot(2, 2, 2)
    plt.axis('off')
    plt.title('K=2')
    plt.imshow(kmeans2)
    io.imsave('kmeans2.png', kmeans2)

    plt.subplot(2, 2, 3)
    plt.axis('off')
    plt.title('K=3')
    plt.imshow(kmeans3)
    io.imsave('kmeans3.png', kmeans3)

    plt.subplot(2, 2, 4)
    plt.axis('off')
    plt.title('K=4')
    plt.imshow(kmeans4)
    io.imsave('kmeans4.png', kmeans4)

    plt.show()