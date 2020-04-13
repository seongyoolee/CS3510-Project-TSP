import sys
import numpy as np
import time
import math
import multiprocessing

from termcolor import colored


def calc_dist(num_nodes, x_vals, y_vals):
    matrix = np.ndarray((num_nodes, num_nodes), int)
    for i in range(num_nodes):
        matrix[i][i] = 0
        for j in range(i + 1, num_nodes):
            # check if it can be improved with numpy euclidian distance
            matrix[i][j] = matrix[j][i] = round(math.sqrt((x_vals[i] - x_vals[j]) ** 2 + (y_vals[i] - y_vals[j]) ** 2))
    return matrix

class TSP:
    def __init__(self):
        self.adj_mat = [[]]
        self.possible_values = []

    def tsp(self):
        # Read input file
        f = open(sys.argv[1], "r")
        split_val = f.read().split()
        node_set = set(split_val[::3])
        x_vals = list(map(float, split_val[1::3]))
        y_vals = list(map(float, split_val[2::3]))

        self.adj_mat = calc_dist(len(node_set), x_vals, y_vals)
        print(self.adj_mat)

        # start at node 1
        return self.tsp_helper("1", node_set)

    def tsp_helper(self, node, node_set):
        if not node_set:
            return self.adj_mat[int(node) - 1][0]
        else:
            node_set.remove(node)
            for next_node in node_set:
                print(colored(node_set, "yellow"))
                edge = self.adj_mat[int(node) - 1][int(next_node) - 1]
                print(colored(("selected edge is: ", edge), "blue"))

                rest = self.tsp_helper(next_node, node_set)
                self.possible_values.append(edge + rest)
                # print(colored(self.possible_values, "red"))
            print(colored(self.possible_values, "red"))
            min_case = min(self.possible_values)
            self.possible_values = []
            return min_case

if __name__ == '__main__':

    # setup process
    p = multiprocessing.Process(target=TSP().tsp())

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

    # create a new file w/ algorithm results
    f= open(sys.argv[2], "w+")
    for i in range(10):
         f.write("This is line %d\r\n" % (i+1))
    f.close()

    # print processing time
    print(colored(str(time.time() - start_time), "yellow"))
