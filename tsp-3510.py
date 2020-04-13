import sys
import numpy as np
import time
import math
import multiprocessing
import pandas as pd
import matplotlib.pyplot as plt
from termcolor import colored


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

class TSP:
    def __init__(self, nodes, network, population_size):
        self.nodes = nodes
        self.network = network
        self.population_size = population_size

    def find_closest(self, points, point):
        # distances = np.linalg.norm(nodes - node, axis=1)
        return self.find_distance(points, point).argmin()

    def find_distance(self, p1, p2):
        return np.linalg.norm(p1 - p2, axis=1)

    def get_neighbor_nodes(self, center, radix, domain):
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

    def get_route(self, nodes, network):
        nodes['closest'] = nodes[['x', 'y']].apply(lambda e: self.find_closest(network, e), axis=1, raw=True)
        # route = nodes.sort_values('closest')["node"].values.tolist()
        # start_index = route.index('1')
        # return route[start_index:] + route[:start_index]
        return nodes.sort_values('closest').index

    def tsp(self):

        # set learning rate
        learning_rate = 0.8

        # set number of iterations
        for iteration in range(100000):
            node_coord = self.nodes.sample(1)[['x', 'y']].values

            closest_index = self.find_closest(self.network, node_coord)

            gaussian = self.get_neighbor_nodes(closest_index, int(self.population_size / 10), self.network.shape[0])

            network_copy = self.network.copy()
            self.network += np.reshape(gaussian, (-1, 1)) * learning_rate * (node_coord - self.network)

            # # show the change in network applying gaussian
            # plt.plot(network_copy[:,0], network_copy[:,1], "or")
            # plt.plot(network[:,0], network[:,1], "ob")
            # plt.show()

            # update learning rate & population size
            learning_rate = learning_rate * 0.99997
            self.population_size = self.population_size * 0.9997

            if self.population_size < 1 or learning_rate < 0.0001:
                # compute finished
                break

        # print(colored(nodes, "yellow"))
        # print(colored(network, "blue"))

        # fig, axs = plt.subplots(2)
        # axs[0].plot(nodes.x, nodes.y, "or")
        # axs[1].plot(network[:, 0], network[:, 1], "ob", markersize=2)
        # plt.show()


if __name__ == '__main__':

    # process input file
    nodes, original = process_input(sys.argv[1])

    # network of 8 * num_cities in tour
    population_size = nodes.shape[0] * 8
    network = np.random.rand(population_size, 2)

    tsp_obj = TSP(nodes, network, population_size)

    # record start time
    start_time = time.time()
    # start_time2 = time.time()

    # tsp()

    # setup process
    # p = multiprocessing.Process(target=tsp, args=(nodes, network, population_size))
    p = multiprocessing.Process(target=tsp_obj.tsp)

    # start tsp
    p.start()

    # wait for <time> or until process finishes
    p.join(timeout=int(sys.argv[3]))

    # terminate
    if p.is_alive():
        print("terminating")
        # Terminate
        p.terminate()
        p.join()

    route_index = tsp_obj.get_route(tsp_obj.nodes, tsp_obj.network)
    original = original.reindex(route_index)
    route = original['node'].values.tolist()
    route = np.roll(original['node'], -(route.index('1'))).tolist()
    route.append('1')
    print(route)
    distances = tsp_obj.find_distance(original[['x', 'y']], np.roll(original[['x', 'y']], 1, axis=0))
    distance = np.sum(distances)
    print(distance)

    answer = ['1', '2', '6', '10', '11', '12', '15', '19', '18', '17', '21', '22', '23', '29', '28', '26', '20', '25', '27', '24', '16', '14', '13', '9', '7', '3', '4', '8', '5', '1']
    print((np.array_equal(answer, route)) or (np.array_equal(route, np.flip(answer))))


    # write results
    f = open(sys.argv[2], "w+")
    f.write("%f\n" % distance)
    f.writelines(map(lambda e: e + ' ', route))
    f.close()

    # print processing time
    print(colored(str(time.time() - start_time), "yellow"))
    # print(colored(str(time.time() - start_time2), "yellow"))
