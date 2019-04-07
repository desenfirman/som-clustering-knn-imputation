from math import isnan
from math import sqrt
from pandas import DataFrame as df


def impute_dataset(input_dataset, K):
    imputed_dataset = list()
    for row in range(0, len(input_dataset)):
        imputed_dataset_attr = list()
        for attr in range(0, len(input_dataset[row])):
            new_val = input_dataset[row][attr]
            if isnan(input_dataset[row][attr]):
                sorted_dist = get_sorted_distance_group(
                    input_dataset, (row, attr))

                sum_imp = 0
                for idk in range(0, K):
                    b = sorted_dist[idk][1]
                    sum_imp += input_dataset[b][attr]
                new_val = sum_imp / K
            imputed_dataset_attr.append(new_val)
        imputed_dataset.append(imputed_dataset_attr)

    return imputed_dataset


def get_sorted_distance_group(input_dataset, index_data):
    row = index_data[0]
    attr = index_data[1]
    pair_dist_list = list()
    for x in range(0, len(input_dataset)):
        if not isnan(input_dataset[x][attr]):
            pair_dist_list.append((row, x))

    dist = list()
    inverted_input_data = list(map(list, zip(*input_dataset)))
    for a, b in pair_dist_list:
        sum_sq = 0
        for id_attr in range(0, len(input_dataset[0])):
            attr_df = df(inverted_input_data[id_attr])

            max_attr = attr_df.max(skipna=True)
            min_attr = attr_df.min(skipna=True)
            val_a = input_dataset[a][id_attr]
            val_b = input_dataset[b][id_attr]
            sq_dist = 1 if (isnan(val_a) or isnan(
                val_b)) else ((abs(val_b - val_a) / (
                    max_attr - min_attr)) ** 2)
            sum_sq += sq_dist
        euc_dist = sqrt(sum_sq)
        dist.append((a, b, euc_dist))
    dist_sorted = sorted(dist, key=lambda x: x[2])
    return dist_sorted
