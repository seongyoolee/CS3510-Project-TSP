import sys
import numpy as np
import time
import math
import multiprocessing

from termcolor import colored

def calc_dist(num_nodes, x_vals, y_vals):
    matrix = np.ndarray((num_nodes, num_nodes), int)
    for i in range(num_nodes):
        for j in range(i, num_nodes):
            matrix[i][j] = matrix[j][i] = int(math.sqrt(x_vals[i] ** 2 + y_vals[j] ** 2))
    return matrix

def tsp(input_file):
    # Read input file
    f = open(sys.argv[1], "r")
    split_val = f.read().split()
    nodes = split_val[::3]
    x_vals = list(map(float, split_val[1::3]))
    y_vals = list(map(float, split_val[2::3]))

    adj_mat = calc_dist(len(nodes), x_vals, y_vals)
    # print(adj_mat)

if __name__ == '__main__':

    # setup process
    p = multiprocessing.Process(target=tsp(sys.argv[1]))

    # record start time
    start_time = time.time()

    # start tsp
    p.start()

    # wait for <time> or until process finishes
    p.join(int(sys.argv[3]))

    # terminate
    if p.is_alive():
        # save to tour.txt here?

        print("terminating")

        # Terminate
        p.terminate()
        p.join()

    # print processing time
    print(colored(str(time.time() - start_time), "yellow"))
