import sys
import numpy as np
import multiprocessing
import pandas as pd
# from termcolor import colored

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
    deltas = np.absolute(mu - np.arange(sample_size))
    # print(colored(deltas, "blue"))

    # circular network
    distances = np.minimum(deltas, sample_size - deltas)
    # print(colored(distances, "yellow"))

    sigma = 1 if sigma < 1 else sigma
    gaussian_distribution = np.exp(-0.5 * ((distances * distances) / (sigma * sigma)))
    # print(colored(gaussian_distribution, "red"))

    return gaussian_distribution

def get_route(nodes, network):
    nodes['closest'] = nodes[['x', 'y']].apply(lambda e: find_closest(network, e), axis=1, raw=True)
    sorted_nodes = nodes.sort_values('closest')
    return sorted_nodes.index

def tsp(manager_list):
    # set learning rate
    learning_rate = 0.8

    while True:

        # pick random node
        node_coord = manager_list[0].sample(1)[['x', 'y']].values

        # get the closest neighbor in network
        closest_index = find_closest(manager_list[1], node_coord)

        # get gaussian distribution on closest neighbor
        gaussian = get_neighbor_neurons(closest_index, int(manager_list[2] / 10), manager_list[1].shape[0])

        # update network based on gaussian
        manager_list[1] += np.reshape(gaussian, (-1, 1)) * learning_rate * (node_coord - manager_list[1])

        # update learning rate & population size
        learning_rate = learning_rate * 0.99997
        manager_list[2] = manager_list[2] * 0.9997

        if manager_list[2] < 1 or learning_rate < 0.0001:
            # compute finished
            break


if __name__ == '__main__':

    # process input file
    nodes = process_input(sys.argv[1])

    distance_list, route_list = [], []
    for i in range(10):
        # neuron network of 8 * num_nodes in tour
        neuron_network_size = nodes.shape[0] * 8
        network = np.random.rand(neuron_network_size, 2)
        max_val = nodes[['x', 'y']].max()
        min_val = nodes[['x', 'y']].min()
        x_range = max_val.x - min_val.x
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
        p.start()

        # wait for <time> or until process finishes
        p.join(timeout=int(sys.argv[3]))

        # terminate
        if p.is_alive():
            # print("terminating")
            p.terminate()
            p.join()

        # get values from shared variable
        network = manager_list[1]

        # get route
        route_index = get_route(nodes, network)
        nodes = nodes.reindex(route_index)
        route = nodes['node'].values.tolist()
        route = np.roll(nodes['node'], -(route.index('1'))).tolist()
        route.append('1')
        # print(colored(route, "blue"))
        route_list.append(route)

        # get distance
        distances = find_distance(nodes[['x', 'y']], np.roll(nodes[['x', 'y']], 1, axis=0))
        distance = np.sum(distances)
        # print(colored(distance, "green"))
        distance_list.append(distance)

    # write results
    f = open(sys.argv[2], "w+")
    for i in range(10):
        f.write("%d\n" % distance_list[i])
        f.writelines(map(lambda e: e + ' ', route_list[i]))
        f.write("\n")
    f.write("Average Distance: %s\n" % np.average(distance_list))
    f.write("Standard Deviation: %s" % np.std(distance_list))
    f.close()