import numpy as np


def distance(x, y, p=2):
    size = len(x)
    sum_x = 0
    if size != len(y):
        return 0
    for i in range(size):
        sum_x += pow((x[i] - y[i]), p)

    return pow(sum_x, 1 / p)


class NearestNeighbor:
    def __init__(self):
        self.data_list = []
        self.data_label = []

    def print_data(self):
        for i in range(len(self.data_list)):
            print("Label: ", self.data_label[i], " data: ", self.data_list[i])

    def read_data(self, file_name):
        file = open(file_name, 'r')
        data_row = []
        for row in file.read().split('\n'):
            count = 0
            data_row.clear()
            for x in row.split(' '):
                if x != ' ' and x != '' and count != 0:
                    data_row.append(float(x))
                    count += 1
                if x != ' ' and x != '' and count == 0:
                    self.data_label.append(float(x))
                    count += 1
            if len(data_row) > 0:
                self.data_list.append(np.array(data_row))

    def KNN(self, test_data, feature, K=1):
        distance_list = list()
        neighbors = list()
        for i in range(len(self.data_list)):
            training_data = self.make_test_data(feature, i)
            dist = distance(test_data, training_data)
            distance_list.append(training_data, dist, self.data_label[i])

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

        return unique_label_[index]

    def accuracy(self, test_labels):
        number_of_data = len(self.data_list)
        correct_count = 0
        for i in number_of_data:
            if test_labels[i] == self.data_label[i]:
                correct_count += 1

        return correct_count / number_of_data

    def make_test_data(self, lst, index):
        test_data = []
        for i in lst:
            test_data.append(self.data_list[index][i])
        return test_data

    def forward_selection(self, sLevel=0.5):
        best_feature = []
        feature_number = len(self.data_list[0])
        init_feature = []
        for i in range(feature_number):
            init_feature.append(i)
        test_label = []
        acc_list = []
        test_data = []
        while len(init_feature) > 0:
            test_label.clear()
            acc_list.clear()
            if len(best_feature) == 0:
                for i in init_feature:
                    for j in range(len(self.data_list)):
                        test_data.append(self.data_list[j][i])
                        feature = [i]
                        test_label.append(self.KNN(test_data, feature))
                    acc = (i, self.accuracy(test_label))
                    acc_list.append(acc)
                acc_list.sort(reverse=True, key=lambda pair: pair[1])
            else:
                for i in range(feature_number):
                    for j in range(len(self.data_list)):
                        feature = best_feature
                        feature.append(i)
                        test_data = self.make_test_data(best_feature, j)
                        test_data.append(self.data_list[j][i])
                        test_label.append(self.KNN(test_data, feature))
                    acc = (i, self.accuracy(test_label))
                    acc_list.append(acc)
                acc_list.sort(reverse=True, key=lambda pair: pair[1])

            best_feature.append(acc_list[0][0])
            init_feature.remove(acc_list[0][0])
            if 1 - acc_list[0][1] < sLevel:
                break

        return best_feature

    def backward_elimination(self, target, sLevel=0.5):
        best_feature = []
        init_feature = self.data_list

        return best_feature
