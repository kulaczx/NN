# import here
import time
from nearestN import NearestNeighbor


def menu():
    NN = NearestNeighbor()
    file_name = ''
    alg_num = 0
    print("Welcome to Bertie Woosters Feature Selection Algorithm")
    print("Type in the name of the file to test: ", end='')
    input(file_name)
    NN.read_data(file_name)
    print("Type the number of the algorithm you want to run.")
    print("1) Forward Selection")
    print("2) Backward Elimination")
    input(alg_num)

    if alg_num == 1:
        # run forward
        timer = time.time()
        result = NN.forward_selection()
        endTime = time.time()
        total_time = endTime - timer
        print("Result of Forward Selection is: ", result, " eclipse time: ", total_time)
    if alg_num == 2:
        # run backward
        timer = time.time()
        result = NN.backward_elimination()
        endTime = time.time()
        total_time = endTime - timer
        print("Result of Backward Elimination is: ", result, " eclipse time: ", total_time)


large_data = "Ver_2_CS170_Fall_2021_LARGE_data__82.txt"
small_data = "Ver_2_CS170_Fall_2021_Small_data__43.txt"
nn = NearestNeighbor()
nn.read_data(small_data)
nn.print_data()
#f = nn.forward_selection()
#print(f)
b = nn.backward_elimination()