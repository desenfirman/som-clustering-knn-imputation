from math import isnan
from math import sqrt
from pandas import DataFrame as df
from multiprocessing import Pool


def impute_dataset(input_dataset, K):
    imputed_dataset = list()
    pair_dist_list = get_dist_pair(input_dataset)
    # print(pair_dist_list)
    for row in range(0, len(input_dataset)):
        filtered_pair_dist_list = [
            (x, y, dist) if x == row else (y, x, dist)
            for (x, y, dist) in pair_dist_list if x == row or y == row
        ]
        sorted_dist = sorted(filtered_pair_dist_list, key=lambda x: x[2])
        imputed_dataset_attr = list()
        for attr in range(0, len(input_dataset[row])):
            new_val = input_dataset[row][attr]
            if isnan(input_dataset[row][attr]):
                if K == 0:
                    imputed_dataset_attr.append(0)
                    continue

                sum_imp = 0
                count = 0
                for a, b, dist in sorted_dist:
                    # wk = 1 / (dist ** 2)
                    if isnan(input_dataset[b][attr]):
                        continue
                    sum_imp += (input_dataset[b][attr])
                    count += 1
                    if count == K:
                        break
                new_val = sum_imp / K
            imputed_dataset_attr.append(new_val)
        imputed_dataset.append(imputed_dataset_attr)

    return imputed_dataset


def get_dist_pair(input_dataset):
    pair_dist_list = list()
    for i in range(0, len(input_dataset)):
        for j in range(0, len(input_dataset)):
            if ((i != j) and
                    ((i, j) not in pair_dist_list) and
                    ((j, i) not in pair_dist_list)):
                    pair_dist_list.append((i, j))

    with Pool() as pool:
        dist = pool.starmap(get_euclidian_dist, [
                            (input_dataset, a, b) for a, b in pair_dist_list])
        pool.close()
        pool.join()
    return dist


def get_euclidian_dist(input_dataset, a, b):
    inverted_input_data = list(map(list, zip(*input_dataset)))
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
    return (a, b, euc_dist)
