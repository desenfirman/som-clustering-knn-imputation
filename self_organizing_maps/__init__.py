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
