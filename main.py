# import here
import time
from nearestN import NearestNeighbor
import matplotlib.pyplot as plt
import pandas as pd


def menu():
    NN = NearestNeighbor()
    file_name = ''
    alg_num = 0
    print("Welcome to Bertie Woosters Feature Selection Algorithm")

    while file_name != 'e':
        file_name = input("Type in the name of the file to test: ")
        if '.txt' in file_name:
            NN.read_data(file_name)
            print("Type the number of the algorithm you want to run.")
            print("1) Forward Selection")
            print("2) Backward Elimination")
            alg_num = input()

            if alg_num == '1':
                # run forward
                timer = time.time()
                result, overall = NN.forward_selection()
                endTime = time.time()
                total_time = endTime - timer
                print("Result of Forward Selection is: ", result, " eclipse time: ", total_time)
                df = pd.DataFrame(overall, columns=['Feature', 'Accuracy'])
                df.plot(kind='bar', x='Feature')
                plt.show()

            if alg_num == '2':
                # run backward
                timer = time.time()
                result, overall = NN.backward_elimination()
                endTime = time.time()
                total_time = endTime - timer
                for i in overall:
                    print(i)
                print("Result of Backward Elimination is: ", result, " eclipse time: ", total_time)

                df = pd.DataFrame(overall, columns=['Feature', 'Accuracy'])
                df.plot(kind='bar', x='Feature')
                plt.show()


large_data = "Ver_2_CS170_Fall_2021_LARGE_data__82.txt"
small_data = "Ver_2_CS170_Fall_2021_Small_data__43.txt"

menu()