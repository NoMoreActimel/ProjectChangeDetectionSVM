import numpy as np
import pandas as pd
from sklearn import svm 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import MinMaxScaler # normalization
import matplotlib.pyplot as plt # plotting
from matplotlib import cm
import seaborn # plottin
from PIL import Image # image opening
import math
import os


# PLOTTING
def plot_array_as_image(X, m, n, is_1d=False):
    if is_1d:
        X = X.reshape((m, n))
    plt.figure(figsize=(8, 8))
    plt.imshow(X)

def plot_array_as_image_color(X, m, n, is_1d=False):
    if is_1d:
        X = np.array([np.linalg.norm(pixel_diff) for pixel_diff in X])
        X = X.reshape((m, n))

    plt.figure(figsize=(8, 8))
    plt.imshow(X)


def plot_image(Im):
    plt.figure(1, (8, 8))
    plt.imshow(Im)

    
def plot_deltas(count_deltas, p_value_ind):
    plt.figure(figsize=(8, 8))

    plt.plot(count_deltas, label='deltas based on frobenius norm', color='b')
    plt.axvline(x=p_value_ind, label='p_value_ind', color='r')
    plt.legend()
    plt.tight_layout()


class Data:
    # IMAGE SCANNING
    def scan_img_color(self, path_img1, path_img2):
        self.img1 = list()
        self.img2 = list()
        self.img1.append(Image.open(path_img1))
        self.img2.append(Image.open(path_img2))

        # take the difference between two images
        difference = np.array(self.img1[0], dtype=np.int16) - np.array(self.img2[0], dtype=np.int16)
        self.m, self.n, self.dim = difference.shape
        self.samples = self.m * self.n
        # reshape it
        difference = difference.reshape((self.samples, self.dim), order='C')
        # normalize them and save in X
        self.X = MinMaxScaler().fit_transform(difference)
        # now count norms for each pixel's difference
        self.norms = np.array([np.linalg.norm(pixel_diff) for pixel_diff in self.X])

        self.y = np.full(self.samples, False, dtype=bool)
        self.classify_changes(p_value=0.6, show_deltas=False, is_bw=False)
        plot_array_as_image(self.y, self.m, self.n, is_1d=True)

    def scan_img_bw(self, path_img1, path_img2):
        self.img1 = list()
        self.img2 = list()
        self.img1.append(Image.open(path_img1).convert('L'))
        self.img2.append(Image.open(path_img2).convert('L'))

        # take the difference between two images
        difference = np.array(self.img1[0], dtype=np.int16) - np.array(self.img2[0], dtype=np.int16)
        self.m, self.n = difference.shape
        self.samples = self.m * self.n
        # reshape it
        difference = difference.reshape((self.samples, 1), order='C')
        # normalize them and save in X
        self.X = MinMaxScaler().fit_transform(difference)
        self.norms = self.X

        # LABELS PART
        self.y = np.full(self.samples, False, dtype=bool)
        self.classify_changes(p_value=0.6, show_deltas=False, is_bw=True)
        plot_array_as_image(self.y, self.m, self.n, is_1d=True)
        plot_array_as_image(self.X, self.m, self.n, is_1d=True)

    def scan_dir_imgs_color(self, path_images1, path_images2, path_labels, m=512, n=512, dim=3, cnt_images=1, start_from=0):
        # suppose images are named the same in all 3 directories
        # We will use each pixel as a sample with dim features
        # Storing samples in X and labels in y

        img_filenames = os.listdir(path_images1)

        self.m, self.n, self.features = m, n, dim
        self.img_size = self.m * self.n
        self.cnt_images = cnt_images
        self.samples = cnt_images * self.img_size
        self.X = np.array([np.full(self.features, 0.0, dtype=np.float64) for i in range(self.samples)])
        self.y = np.full(self.samples, 0, dtype=np.int32)

        self.img1 = list()
        self.img2 = list()

        for im_ind in range(start_from, len(img_filenames)):
            im_offset = im_ind - start_from
            if im_offset % 10 == 0 or im_offset == cnt_images - 1:
                print("image no.", im_ind)
                print(img_filenames[im_ind])

            # IMAGES PART
            self.img1.append(Image.open(path_images1 + "/" + img_filenames[im_ind]))
            self.img2.append(Image.open(path_images2 + "/" + img_filenames[im_ind]))

            # take the difference between two images
            difference = np.array(self.img1[im_offset], dtype=np.int16) - np.array(self.img2[im_offset], dtype=np.int16)
            # reshape it
            difference = difference.reshape((self.img_size, self.features), order='C')
            # normalize them
            difference = MinMaxScaler().fit_transform(difference)

            # save pixel norms as features in X[im_ind]
            for pixel_ind in range(self.img_size):
                self.X[im_offset * self.img_size + pixel_ind] = difference[pixel_ind]
        
            # LABELS PART
            img_labels = Image.open(path_labels + "/" + img_filenames[im_ind]).convert('L')    
            pixel_labels = np.array(img_labels, dtype=np.int16).reshape((self.img_size, 1))

            # store bool(pixel_label)
            for pixel_ind in range(self.img_size):
                self.y[im_offset * self.img_size + pixel_ind] = np.sign(pixel_labels[pixel_ind] - 255)
        
            # stop criteria
            if im_offset == cnt_images - 1:
                break

    def scan_dir_imgs_bw(self, path_images1, path_images2, path_labels, m=512, n=512, cnt_images=1, start_from=0):
        # suppose images are named the same in all 3 directories
        # We will use each pixel as a sample with dim features
        # Storing samples in X and labels in y

        img_filenames = os.listdir(path_images1)

        self.m, self.n, self.features = m, n, 1
        self.img_size = self.m * self.n
        self.cnt_images = cnt_images
        self.samples = cnt_images * self.img_size
        self.X = np.array([np.full(self.features, 0.0, dtype=np.float64) for i in range(self.samples)])
        self.y = np.full(self.samples, 0, dtype=np.int32)

        self.img1 = list()
        self.img2 = list()

        for im_ind in range(start_from, len(img_filenames)):
            im_offset = im_ind - start_from
            if im_offset % 10 == 0 or im_offset == cnt_images - 1:
                print("image no.", im_ind)
                print(img_filenames[im_ind])

            # IMAGES PART
            self.img1.append(Image.open(path_images1 + "/" + img_filenames[im_ind]))
            self.img2.append(Image.open(path_images2 + "/" + img_filenames[im_ind]))

            # take the difference between two images
            difference = np.array(self.img1[im_offset].convert('L'), dtype=np.int16) - np.array(
                self.img2[im_offset].convert('L'), dtype=np.int16)
            # reshape it
            difference = difference.reshape((self.img_size, self.features), order='C')
            # normalize them
            difference = MinMaxScaler().fit_transform(difference)

            # save pixel norms as features in X[im_ind]
            for pixel_ind in range(self.img_size):
                self.X[im_offset * self.img_size + pixel_ind] = difference[pixel_ind]
        
            # LABELS PART
            img_labels = Image.open(path_labels + "/" + img_filenames[im_ind]).convert('L')    
            pixel_labels = np.array(img_labels, dtype=np.int16).reshape((self.img_size, 1))

            # store bool(pixel_label)
            for pixel_ind in range(self.img_size):
                self.y[im_offset * self.img_size + pixel_ind] = np.sign(pixel_labels[pixel_ind] - 255)
        
            # stop criteria
            if im_offset == cnt_images - 1:
                break


    # CLUSTERISATION
    def count_cluster(self, data, clusters, img_ind, ind1, ind2):
        img_offset_clusters = img_ind * self.clusters_per_image
        img_offset_data = img_ind * self.img_size

        for i in range(self.cluster_size * ind1, self.cluster_size * ind1 + self.cluster_size):
            for j in range(self.cluster_size * ind2, self.cluster_size * ind2 + self.cluster_size):
                clusters[img_offset_clusters + ind1 * self.n_clusters + ind2] += data[img_offset_data + i * self.n + j]
        clusters[img_offset_clusters + ind1 * self.n_clusters + ind2] /= self.clusters_per_image


    # assuming the image is n * n pixels and has 1 feature
    # and cluster_size divides n
    def clusterise_data(self, cluster_size=16):
        self.cluster_size = cluster_size
        self.n_clusters = int(self.n / cluster_size)
        self.clusters_per_image = self.n_clusters * self.n_clusters

        self.X_clusters = np.full((self.cnt_images * self.n_clusters * self.n_clusters, 1), 0.0, dtype=np.float64)
        self.y_clusters = np.full(self.cnt_images * self.n_clusters * self.n_clusters, 0.0, dtype=np.float64)
        
        for img_ind in range(self.cnt_images):
            for i in range(self.n_clusters):
                for j in range(self.n_clusters):
                    self.count_cluster(self.X, self.X_clusters, img_ind, i, j)
                    self.count_cluster(self.y, self.y_clusters, img_ind, i, j)
                    self.y_clusters = np.sign(self.y_clusters)
    
    
    def count_cluster_dim(self, data, clusters, img_ind, ind1, ind2):
        img_offset_clusters = img_ind * self.clusters_per_image
        img_offset_data = img_ind * self.img_size

        for i in range(self.cluster_size * ind1, self.cluster_size * ind1 + self.cluster_size):
            for j in range(self.cluster_size * ind2, self.cluster_size * ind2 + self.cluster_size):
                for f in range(self.features):
                    clusters[img_offset_clusters + ind1 * self.n_clusters + ind2][f] += data[img_offset_data + i * self.n + j][f]
        
        for f in range(self.features):
            clusters[img_offset_clusters + ind1 * self.n_clusters + ind2][f] /= self.clusters_per_image

    # assuming the image is n * n pixels and has 1 feature
    # and cluster_size divides n
    def clusterise_data_dim(self, cluster_size=16):
        self.cluster_size = cluster_size
        self.n_clusters = int(self.n / cluster_size)
        self.clusters_per_image = self.n_clusters * self.n_clusters

        self.X_clusters = np.full((self.cnt_images * self.n_clusters * self.n_clusters, self.features), 0.0, dtype=np.float64)
        self.y_clusters = np.full(self.cnt_images * self.n_clusters * self.n_clusters, 0.0, dtype=np.float64)
        
        for img_ind in range(self.cnt_images):
            for i in range(self.n_clusters):
                for j in range(self.n_clusters):
                    self.count_cluster_dim(self.X, self.X_clusters, img_ind, i, j)
                    self.count_cluster(self.y, self.y_clusters, img_ind, i, j)
                    self.y_clusters = np.sign(self.y_clusters)

    # def drop_p_values(self, p_value):
    #     for im_ind in self.cnt_images:
            


class SVM_Change_Detection_Classifier:
    def get_images(self, path_im1, path_im2, path_labels, cnt_train_images, 
        cnt_test_images, start_from_train=0, start_from_test=100, color=False, cluster_size=8):

        self.D_train = Data()
        if color:
            self.D_train.scan_dir_imgs_color(path_im1, path_im2, path_labels, 
                cnt_images=cnt_train_images, start_from=start_from_train)
        else:
            self.D_train.scan_dir_imgs_bw(path_im1, path_im2, path_labels,
                cnt_images=cnt_train_images, start_from=start_from_train)
        
        self.D_train.clusterise_data(cluster_size=cluster_size)

        self.D_test = Data()
        if color:
            self.D_test.scan_dir_imgs_color(path_im1, path_im2, path_labels, 
                cnt_images=cnt_test_images, start_from=start_from_test)
        else:
            self.D_test.scan_dir_imgs_bw(path_im1, path_im2, path_labels, 
                cnt_images=cnt_test_images, start_from=start_from_test)
        
        self.D_test.clusterise_data(cluster_size=cluster_size)

    def train(self):
        self.clf = svm.SVC(kernel='rbf', gamma='scale', C=0.1)
        self.clf.fit(self.D_train.X_clusters, self.D_train.y_clusters)
    
    def correct_labels(self, test_image_ind=0):
        i = test_image_ind * self.D_test.clusters_per_image
        j = (test_image_ind + 1) * self.D_test.clusters_per_image
        return self.D_test.y_clusters[i:j]
    
    def predict_labels(self, test_image_ind=0):
        i = test_image_ind * self.D_test.clusters_per_image
        j = (test_image_ind + 1) * self.D_test.clusters_per_image
        return self.clf.predict(self.D_test.X_clusters[i:j])

    def get_initial_image(self, test_image_ind=0):
        return self.D_test.img1[test_image_ind]
    
    def get_final_image(self, test_image_ind=0):
        return self.D_test.img2[test_image_ind]


path_im1 = './data/SECOND_train_set/im1/'
path_im2 = './data/SECOND_train_set/im2/'
path_labels = './data/SECOND_train_set/label1/'


def make_D(X1, X2):
    m, n, dim = X1.shape
    X_dif = X1 - X2
    D_cur = Data()
    D_cur.m, D_cur.n = m, n
    D_cur.X = np.zeros((m, n), dtype=np.float64)
    D_cur.y = np.zeros((m, n))
    D_cur.cnt_images = 1
    D_cur.img_size = m * n

    for i in range(m):
        for j in range(n):
            D_cur.X[i][j] = np.linalg.norm(X_dif[i][j])
    D_cur.X = D_cur.X.reshape((m * n, 1))
    D_cur.X = MinMaxScaler().fit_transform(D_cur.X)
    D_cur.y = D_cur.y.reshape((m * n, 1))
    
    D_cur.clusterise_data(cluster_size=16)
    return D_cur

def predict(X1, X2):
    S_ = SVM_Change_Detection_Classifier()
    S_.get_images(path_im1, path_im2, path_labels, cnt_train_images=16, cnt_test_images=1,
        start_from_train=0, start_from_test=120, cluster_size=16)
    S_.train()
    
    D_cur = make_D(X1, X2)
    y_predicted = S_.clf.predict(D_cur.X_clusters)
    plot_array_as_image(D_cur.X_clusters, m=32, n=32, is_1d=True)

    for i in range(y_predicted.shape[0]):
        if y_predicted[i] == -1:
            y_predicted[i] = 0
        else:
            y_predicted[i] = 1
    return y_predicted

X1 = np.array(Image.open(path_im1 + "/" + "00918.png"), dtype=np.int16)
X2 = np.array(Image.open(path_im2 + "/" + "00918.png"), dtype=np.int16)
plot_array_as_image(predict(X1, X2), m=32, n=32, is_1d=True)