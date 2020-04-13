import sys
import numpy as np
import time
import multiprocessing
import pandas as pd
import matplotlib.pyplot as plt
from termcolor import colored

def process_input(file):
    # read input file
    processed = pd.read_csv(open(file, "r"), sep=' ', names=['node', 'x', 'y'],
                            dtype={'node': str, 'x': np.float64, 'y': np.float64})
    return processed

def find_closest(points, point):
    return find_distance(points, point).argmin()

def find_distance(p1, p2):
    distances = np.round(np.linalg.norm(p1 - p2, axis=1))
    return distances.astype(int)

def get_neighbor_neurons(mu, sigma, sample_size):
    # normal distribution = e ^ (-0.5 * ((x - mu)/sigma)^2)

    deltas = np.absolute(mu - np.arange(sample_size))
    # print(colored(deltas, "blue"))

    distances = np.minimum(deltas, sample_size - deltas)
    # print(colored(distances, "yellow"))

    sigma = 1 if sigma < 1 else sigma
    gaussian_distribution = np.exp(-0.5 * ((distances * distances) / (sigma * sigma)))
    # gaussian_distribution = np.exp(-(distances * distances) / (2 * (sigma * sigma)))
    # print(colored(gaussian_distribution, "red"))

    # fig, axs = plt.subplots(2)
    # axs[0].plot(distances)
    # axs[1].plot(gaussian_distribution)
    # plt.show()
    return gaussian_distribution

def get_route(nodes, network):
    nodes['closest'] = nodes[['x', 'y']].apply(lambda e: find_closest(network, e), axis=1, raw=True)
    return nodes.sort_values('closest').index

def tsp(manager_list):

    # set learning rate
    learning_rate = 0.8

    # set number of iterations
    for iteration in range(100000):

        # pick random node
        node_coord = manager_list[0].sample(1)[['x', 'y']].values

        # get the closest neighbor in network
        closest_index = find_closest(manager_list[1], node_coord)

        # get gaussian distribution on closest neighbor
        gaussian = get_neighbor_neurons(closest_index, int(manager_list[2] / 10), manager_list[1].shape[0])

        # update network based on gaussian
        manager_list[1] += np.reshape(gaussian, (-1, 1)) * learning_rate * (node_coord - manager_list[1])

        # # show the change in network applying gaussian
        # plt.plot(network_copy[:,0], network_copy[:,1], "or")
        # plt.plot(network[:,0], network[:,1], "ob")
        # plt.show()

        # update learning rate & population size
        learning_rate = learning_rate * 0.99997
        manager_list[2] = manager_list[2] * 0.9997

        if manager_list[2] < 1 or learning_rate < 0.0001:
            # compute finished
            break

    # fig, axs = plt.subplots(2)
    # axs[0].plot(nodes.x, nodes.y, "or")
    # axs[1].plot(network[:, 0], network[:, 1], "ob", markersize=2)
    # plt.show()


if __name__ == '__main__':

    # record start time
    start_time = time.time()

    # process input file
    nodes = process_input(sys.argv[1])

    # neuron network of 8 * num_nodes in tour
    neuron_network_size = nodes.shape[0] * 8
    network = np.random.rand(neuron_network_size, 2)
    max_val = nodes[['x', 'y']].max()
    min_val = nodes[['x', 'y']].min()
    x_range = max_val.x - min_val.y
    y_range = max_val.y - min_val.y
    network[:, 0] = network[:, 0] * x_range + min_val.x
    network[:, 1] = network[:, 1] * y_range + min_val.y

    # setup shared variable
    manager = multiprocessing.Manager()
    manager_list = manager.list()
    manager_list.append(nodes)
    manager_list.append(network)
    manager_list.append(neuron_network_size)

    # setup process
    p = multiprocessing.Process(target=tsp, args=[manager_list])

    # start tsp
    start_time2 = time.time()
    p.start()

    # wait for <time> or until process finishes
    p.join(timeout=int(sys.argv[3]))

    # terminate
    if p.is_alive():
        print("terminating")
        # Terminate
        p.terminate()
        p.join()

    print(colored(str(time.time() - start_time2), "green"))

    # get values from shared variable
    network = manager_list[1]

    # get route
    route_index = get_route(nodes, network)
    nodes = nodes.reindex(route_index)
    route = nodes['node'].values.tolist()
    route = np.roll(nodes['node'], -(route.index('1'))).tolist()
    route.append('1')
    # print(colored(route, "green"))

    # get distance
    distances = find_distance(nodes[['x', 'y']], np.roll(nodes[['x', 'y']], 1, axis=0))
    distance = np.sum(distances)
    # print(colored(distance, "green"))

    # # compare with optimal route - MAT-TEST
    # answer = ['1', '2', '6', '10', '11', '12', '15', '19', '18', '17', '21', '22', '23', '29', '28', '26', '20', '25', '27', '24', '16', '14', '13', '9', '7', '3', '4', '8', '5', '1']
    # print((np.array_equal(answer, route)) or (np.array_equal(route, np.flip(answer))))

    # write results
    f = open(sys.argv[2], "w+")
    f.write("%d\n" % distance)
    f.writelines(map(lambda e: e + ' ', route))
    f.close()

    # print processing time
    print(colored(str(time.time() - start_time), "yellow"))

    # Things that can be changed
    # number of iterations, size of neuron network, initial learning rate, rate of change in learning rate, rate of change in neuron network size