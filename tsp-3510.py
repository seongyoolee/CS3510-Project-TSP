import sys
import numpy as np
import time
import math
import multiprocessing
import pandas as pd

import matplotlib.pyplot as plt

from termcolor import colored

# def calc_dist(num_nodes, x_vals, y_vals):
#     matrix = np.ndarray((num_nodes, num_nodes), int)
#     for i in range(num_nodes):
#         matrix[i][i] = 0
#         for j in range(i + 1, num_nodes):
#             # check if it can be improved with numpy euclidian distance
#             matrix[i][j] = matrix[j][i] = round(math.sqrt((x_vals[i] - x_vals[j]) ** 2 + (y_vals[i] - y_vals[j]) ** 2))
#     return matrix

def process_input(file):
    # read input file
    processed = pd.read_csv(open(file, "r"), sep=' ', names=['node', 'x', 'y'],
                            dtype={'node': str, 'x': np.float64, 'y': np.float64})
    original = processed.copy()

    # normalize
    max_val = processed[['x', 'y']].max()
    min_val = processed[['x', 'y']].min()
    xy_ratio = ((max_val.x - min_val.x) / (max_val.y - min_val.y), 1)
    xy_ratio = np.array(xy_ratio) / max(xy_ratio)

    normalized = processed[['x', 'y']].apply(lambda e: (e - e.min()) / (e.max() - e.min()))
    processed[['x', 'y']] = normalized.apply(lambda e: e * xy_ratio, axis=1)
    return processed, original

def find_closest(nodes, node):
    # distances = np.linalg.norm(nodes - node, axis=1)
    return find_distance(nodes, node).argmin()

def find_distance(nodes, node):
    return np.linalg.norm(nodes - node, axis=1)

def get_neighbor_nodes(center, radix, domain):
    radix = 1 if radix < 1 else radix
    # print(center, radix, domain)
    deltas = np.absolute(center - np.arange(domain))
    # print(colored(deltas, "blue"))
    distances = np.minimum(deltas, domain - deltas)
    # print(colored(distances, "yellow"))
    gaussian_distribution = np.exp(-(distances * distances) / (2 * (radix * radix)))
    # print(colored(gaussian_distribution, "red"))

    # fig, axs = plt.subplots(2)
    # axs[0].plot(distances)
    # axs[1].plot(gaussian_distribution)
    # plt.show()

    return gaussian_distribution

def get_route(nodes, network):
    nodes['closest'] = nodes[['x', 'y']].apply(lambda e: find_closest(network, e), axis=1, raw=True)
    # route = nodes.sort_values('closest')["node"].values.tolist()
    # start_index = route.index('1')
    # return route[start_index:] + route[:start_index]
    return nodes.sort_values('closest').index

def tsp():

    # process input file
    nodes, original = process_input(sys.argv[1])

    # network of 8 * num_cities in tour
    population_size = nodes.shape[0] * 8
    network = np.random.rand(population_size, 2)

    # set learning rate
    learning_rate = 0.8

    # set number of iterations
    for iteration in range(100000):
        node_coord = nodes.sample(1)[['x', 'y']].values

        closest_index = find_closest(network, node_coord)

        gaussian = get_neighbor_nodes(closest_index, int(population_size / 10), network.shape[0])

        network_copy = network.copy()
        network += np.reshape(gaussian, (-1, 1)) * learning_rate * (node_coord - network)

        # # show the change in network applying gaussian
        # plt.plot(network_copy[:,0], network_copy[:,1], "or")
        # plt.plot(network[:,0], network[:,1], "ob")
        # plt.show()

        # update learning rate & population size
        learning_rate = learning_rate * 0.99997
        population_size = population_size * 0.9997

        if population_size < 1 or learning_rate < 0.0001:
            # compute finished
            break

    # print(colored(nodes, "yellow"))
    # print(colored(network, "blue"))

    # fig, axs = plt.subplots(2)
    # axs[0].plot(nodes.x, nodes.y, "or")
    # axs[1].plot(network[:, 0], network[:, 1], "ob", markersize=2)
    # plt.show()

    route_index = get_route(nodes, network)

    original = original.reindex(route_index)
    route = original['node'].values.tolist()
    route = np.roll(original['node'], -(route.index('1'))).tolist()
    route.append('1')
    print(route)
    distances = find_distance(original[['x', 'y']], np.roll(original[['x', 'y']], 1, axis=0))
    print(np.sum(distances))


if __name__ == '__main__':

    tsp()

    # record start time
    start_time = time.time()


    # start_time2 = time.time()

    # setup process
    # p = multiprocessing.Process(target=tsp())

    # start tsp
    # p.start()

    # wait for <time> or until process finishes
    # p.join(timeout=int(sys.argv[3]))

    # terminate
    # if p.is_alive():
        # save to tour.txt here?

        # print("terminating")
        #
        # # Terminate
        # p.terminate()
        # p.join()


    # create a new file w/ algorithm results
    f = open(sys.argv[2], "w+")
    for i in range(10):
        f.write("This is line %d\r\n" % (i+1))
    f.close()

    # print processing time
    print(colored(str(time.time() - start_time), "yellow"))
    print(colored(str(time.time() - start_time2), "yellow"))
