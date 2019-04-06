import random as rd


def normalize_data(input_data):
    normalized_data = input_data
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


def training(dataset_input, input_weight, epoch, alpha, eta):
    trained_weight = input_weight
    return trained_weight


def penentuan_cluster(trained_weight, input_data_row):
    return (0, 0)


def quantization_error(trained_weight, dataset_input):
    return 0


def davies_bouldin_index(dataset_input, clustering_result, trained_weight):
    return 0
