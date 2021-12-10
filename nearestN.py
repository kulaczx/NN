import numpy as np


def distance(x, y, p=2):
    size = len(x)
    sum_x = 0
    if size != len(y):
        return 0
    for i in range(size):
        sum_x += pow((x[i] - y[i]), p)

    return np.sqrt(sum_x)


def make_feature(feature, remove_number):
    new_feature = []
    for i in feature:
        if i != remove_number:
            new_feature.append(i)
    return new_feature


class NearestNeighbor:
    def __init__(self):
        self.data_list = []
        self.data_label = []
        self.unique_label = [1.0, 2.0]

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
            #print("test data: ", test_data, "target: ", training_data ,"dist: ",dist)
            distance_list.append((training_data, dist, self.data_label[i]))

        distance_list.sort(key=lambda tup: tup[1])
        neighbors_label = list()
        for i in range(K):
            neighbors.append(distance_list[i][0])
            neighbors_label.append(distance_list[i][2])

        label_counts = np.zeros(2)
        for i in range(K):
            for j in range(len(self.unique_label)):
                #print(neighbors_label[i], self.unique_label[j])
                if neighbors_label[i] == self.unique_label[j]:
                    label_counts[j] += 1
        index = 0
        prev = label_counts[0]

        for i in range(len(label_counts)):
            if label_counts[i] > prev:
                index = i
                prev = label_counts[i]
        #print("return ", self.unique_label[index])
        return self.unique_label[index]

    def accuracy(self, test_labels):
        number_of_data = len(self.data_list)
        correct_count = 0
        for i in range(number_of_data):
            #print("test label: ", test_labels[i], "target label: ", self.data_label[i])
            if test_labels[i] == self.data_label[i]:
                correct_count += 1

        return correct_count / number_of_data

    def make_test_data(self, lst, index):
        test_data = []
        for i in lst:
            test_data.append(self.data_list[index][i])
        #print(test_data)
        return test_data

    def forward_selection(self, sLevel=0.05):
        best_feature = []
        feature_number = len(self.data_list[0])
        init_feature = []
        for i in range(feature_number):
            init_feature.append(i)
        test_label = []
        acc_list = []
        test_data = []
        overall_acc = []
        while len(init_feature) > 0:
            test_label.clear()
            acc_list.clear()
            if len(best_feature) == 0:
                for i in init_feature:
                    feature = list([i])
                    print("Feature: ", feature, end='')
                    test_label.clear()
                    for j in range(len(self.data_list)):
                        test_data.clear()
                        test_data.append(self.data_list[j][i])
                        test_label.append(self.KNN(test_data, feature, K=6))
                    acc = (i, self.accuracy(test_label))
                    print(" Accuracy: ", acc[1])
                    print()
                    acc_list.append(acc)
                    overall_acc.append((feature, acc[1]))
                acc_list.sort(reverse=True, key=lambda pair: pair[1])
            else:
                feature = []
                for i in init_feature:
                    feature.clear()
                    feature = list(best_feature)
                    feature.append(i)
                    print("Feature: ", feature)
                    test_label.clear()
                    for j in range(len(self.data_list)):
                        test_data.clear()
                        test_data = self.make_test_data(best_feature, j)
                        test_data.append(self.data_list[j][i])
                        #print(test_data)
                        test_label.append(self.KNN(test_data, feature, K=6))
                    acc = (i, self.accuracy(test_label))
                    print(" Accuracy: ", acc[1])
                    print()
                    acc_list.append(acc)
                    overall = (list(feature), acc[1])
                    overall_acc.append(overall)
                acc_list.sort(reverse=True, key=lambda pair: pair[1])

            best_feature.append(acc_list[0][0])
            init_feature.remove(acc_list[0][0])
            if (1 - acc_list[0][1]) < sLevel:
                break

        return best_feature, overall_acc

    def backward_elimination(self, sLevel=0.05):
        feature_number = len(self.data_list[0])
        feature = []
        for i in range(feature_number):
            feature.append(i)
        test_label = []
        acc_list = []
        test_data = []
        next_feature = []
        overall_acc = []
        for i in self.data_list:
            test_label.append(self.KNN(i, feature, K=5))
        acc_all = self.accuracy(test_label)
        if 1 - acc_all < sLevel:
            return feature
        while len(feature) > 0:
            acc_list.clear()
            for i in feature:
                test_label.clear()
                next_feature.clear()
                next_feature = make_feature(feature, i)
                print("Feature: ", next_feature)
                for j in range(len(self.data_list)):
                    test_data.clear()
                    test_data = self.make_test_data(next_feature, j)
                    test_label.append(self.KNN(test_data, next_feature, K=3))
                acc = (i, self.accuracy(test_label))
                print(" Accuracy: ", acc[1])
                print()
                acc_list.append(acc)
                overall_acc.append((list(next_feature), acc[1]))
            acc_list.sort(reverse=True, key=lambda pair: pair[1])
            print("remaining feature: ", feature)
            print("remove: ", acc_list[0][0])
            feature.remove(acc_list[0][0])
            if (1 - acc_list[0][1]) < sLevel:
                break

        return feature, overall_acc
