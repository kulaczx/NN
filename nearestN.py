import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def distance(x, y, p=2):
    size = len(x)
    sum = 0
    if size != len(y):
        return 0
    for i in range(size):
        sum += pow((x[i] - y[i]), p)

    return pow(sum, 1 / p)


class NearestNeighbor:
    def __init__(self):
        self.data_list = []
        self.data_label = []

    def read_data(self, file_name):
        file = open(file_name, 'r')
        data_row = []
        for row in file.read().split('\n'):
            count = 0
            data_row.clear()
            for x in row.split(' '):
                if x.isdigit() and count != 0:
                    data_row.append(float(x))
                if x.isdigit() and count == 0:
                    self.data_label.append(float(x))
                count += 1
            self.data_list.append(np.array(data_row))

    def KNN(self, test_data, K=1):
        distance_list = list()
        neighbors = list()
        for i in range(len(self.data_list)):
            dist = distance(test_data, self.data_list[i])
            distance_list.append(self.data_list[i], dist, self.data_label[i])

        distance_list.sort(key=lambda tup: tup[1])
        neighbors_label = list()
        for i in range(K):
            neighbors.append(distance_list[i][0])
            neighbors_label.append(distance_list[i][2])

        unique_label_ = np.unique(np.array(self.data_label))
        label_counts = np.zeros(len(unique_label_))
        for i in range(K):
            for j in len(unique_label_):
                if neighbors_label[i] == unique_label_[j]:
                    label_counts[j] += 1
        index = 0
        prev = label_counts[0]

        for i in range(len(label_counts)):
            if label_counts[i] > prev:
                index = i
                prev = label_counts[i]

        return test_data, unique_label_[index]

    def accuracy(self, test_labels):
        number_of_data = len(self.data_list)
        correct_count = 0
        for i in number_of_data:
            if test_labels[i] == self.data_label[i]:
                correct_count += 1

        return correct_count/number_of_data

    def backward_elimination(self, target, sLevel=0.5):
        best_feature = []
        init_feature = self.data_list

        return best_feature

    def forward_selection(self, target, sLevel=0.5):
        best_feature = []
        init_feature = self.data_list

        return best_feature
