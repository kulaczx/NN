#iimport here

def menu():
    file_name = ''
    alg_num = 0
    print("Welcome to Bertie Woosters Feature Selection Algorithm")
    print("Type in the name of the file to test: ", end='')
    input(file_name)

    print("Type the number of the algorithm you want to run.")
    print("1) Forward Selection")
    print("2) Backward Elimination")
    input(alg_num)

    if alg_num == 1:
        # run forward
        print("run forward")
    if alg_num == 2:
        # run backward
        print("run backward")
