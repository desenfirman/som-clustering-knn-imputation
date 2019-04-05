import random as rd


def normalize_data():
    return 0


def init_som_net(row_size, column_size, attribute_size):
    weight = list()
    for i in range(1, row_size):
        weight_row = list()
        for j in range(1, column_size):
            weight_column = list()
            for k in range(1, attribute_size):
                weight_column.append(rd.random())
            weight_row.append(weight_column)
        weight.append(weight_row)
    return weight


def training(dataset_input, input_weight, epoch, alpha, eta):
    new_weight = list()
    return new_weight


def penentuan_cluster(trained_weight, input_data_row):
    return (0, 0)


def quantization_error(trained_weight, dataset_input):
    return 0


def davies_bouldin_index(dataset_input, clustering_result, trained_weight):
    return 0



# ex script
from pandas import DataFrame as df
