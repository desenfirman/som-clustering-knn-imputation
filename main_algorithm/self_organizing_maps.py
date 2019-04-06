import random as rd
from math import exp
from math import sqrt


def normalize_data(input_data):
    # invert input data from data_row * attr to attr * data row
    normalized_data = list()
    inverted_input_data = list(map(list, zip(*input_data)))

    for attr in range(0, len(inverted_input_data)):
        min_attr = min(inverted_input_data[attr])
        max_attr = max(inverted_input_data[attr])
        range_attr = max_attr - min_attr
        attr_list = list()
        for row in range(0, len(inverted_input_data[attr])):
            new_value = (inverted_input_data[attr][row] - min_attr) / (
                range_attr)
            attr_list.append(new_value)

        normalized_data.append(attr_list)

    # invert again to original structure of input data
    normalized_data = list(map(list, zip(*normalized_data)))
    return normalized_data


def init_som_net(row_size, column_size, attribute_size):
    weight = list()
    for i in range(0, row_size):
        weight_row = list()
        for j in range(0, column_size):
            weight_column = list()
            for k in range(0, attribute_size):
                weight_column.append(rd.random())
            weight_row.append(weight_column)
        weight.append(weight_row)
    return weight


def get_dist_data_from_neur(trained_weight, input_data):
    # find distance from data input to neuron
    dist_data_to_neur = list()
    for neur_i in range(0, len(trained_weight)):
        for neur_j in range(0, len(trained_weight[neur_i])):
            sumsq = 0
            for attr in range(0, len(trained_weight[neur_i][neur_j])):
                sumsq += (input_data[
                    attr] - trained_weight[neur_i][neur_j][attr]) ** 2
            rootsumsq = sqrt(sumsq)
            dist_data_to_neur.append(((neur_i, neur_j), rootsumsq))
    return dist_data_to_neur


def training(dataset_input, input_weight, max_epoch, alpha_0, eta_0):
    trained_weight = input_weight

    for t in range(1, max_epoch + 1):
        alpha_t = alpha_0 * (1 / t)
        eta_t = eta_0 * exp(-1 * (t / max_epoch))

        for row in range(0, len(dataset_input)):

            # find best-matching-unit neuron
            dist_data_to_neur = get_dist_data_from_neur(
                trained_weight, dataset_input[row])
            dist_sorted = sorted(dist_data_to_neur, key=lambda x: x[1])
            c = dist_sorted[0][0]  # access tuple of neuron index

            # weight update
            for neur_i in range(0, len(trained_weight)):
                for neur_j in range(0, len(trained_weight[neur_i])):
                    for attr in range(0, len(trained_weight[neur_i][neur_j])):
                        dist_rij_rc = sqrt(
                            (neur_i - c[0])**2 + (neur_j - c[1])**2)
                        hij = alpha_t * exp(-1 * (dist_rij_rc)**2 / (
                            2 * (eta_t**2)))
                        input_to_w_old = (dataset_input[row][
                            attr] - trained_weight[neur_i][neur_j][attr])
                        new_weight = (trained_weight[neur_i][neur_j][
                            attr] + (hij * input_to_w_old))
                        trained_weight[neur_i][neur_j][attr] = new_weight

    return trained_weight


def penentuan_cluster(trained_weight, input_data_row):
    dist_data_to_neur = get_dist_data_from_neur(trained_weight, input_data_row)

    # find best-matching-unit neuron
    dist_sorted = sorted(dist_data_to_neur, key=lambda x: x[1])
    c = dist_sorted[0][0]  # access tuple of neuron index

    return c


def quantization_error(trained_weight, dataset_input):
    sum_dist = 0
    for data_input in dataset_input:
        dist_data_to_neur = get_dist_data_from_neur(trained_weight, data_input)
        dist_sorted = sorted(dist_data_to_neur, key=lambda x: x[1])
        sum_dist += dist_sorted[0][1]  # access distance
    qe = sum_dist / len(dataset_input)

    return qe


def get_pairs_of_cls(clust_size):
        cls_d_list  = []
        
        for x in range(0, self.clust_size):
            for y in range(0, self.clust_size):
                if (x != y):
                    cls_d_list.append((x, y))
                
        return cls_d_list

def davies_bouldin_index(dataset_input, clustering_result, trained_weight):
    for dataset_input 

    return 0
