import random as rd
from math import exp
from math import sqrt
import sys


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


def one_epoch_training(dataset_input, trained_weight, alpha_t, eta_t):
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


def training(dataset_input, input_weight, max_epoch, alpha_0, eta_0):
    trained_weight = input_weight

    for t in range(1, max_epoch + 1):
        alpha_t = alpha_0 * (1 / t)
        eta_t = eta_0 * exp(-1 * (t / max_epoch))
        trained_weight = one_epoch_training(
            dataset_input, trained_weight, alpha_t, eta_t)
        print("Progress: {0}% AVG Silhouette Score:{1}".format(
            (float(t) / max_epoch) * 100, average_silhouette(
                trained_weight, dataset_input)), end='\r', flush=True)

    return trained_weight


def penentuan_cluster(trained_weight, input_data_row):
    dist_data_to_neur = get_dist_data_from_neur(trained_weight, input_data_row)

    # find best-matching-unit neuron
    dist_sorted = sorted(dist_data_to_neur, key=lambda x: x[1])
    c = dist_sorted[0][0]  # access tuple of neuron index

    return c


def distance(dataset_input, data_point_a, data_point_b):
    dist = 0
    for attr in range(0, len(dataset_input[data_point_a])):
        dist += (dataset_input[data_point_a][attr] - dataset_input[
            data_point_b][attr]) ** 2
    return sqrt(dist)


def silhouette(trained_weight, dataset_input):
    cluster_result = dict()
    cls_list = list()
    for idx, x_data in enumerate(dataset_input):
        c = penentuan_cluster(trained_weight, x_data)
        cls_id = str(c[0]) + ';' + str(c[1]) + ';'
        if cls_id not in cls_list:
            cls_list.append(cls_id)
            cluster_result[cls_id] = list()
        cluster_result[cls_id].append(idx)

    silhouette_score_data = dict()
    avg = 0
    if len(cluster_result) == 1:
        silhouette_score_data['avg'] = -9999
        return silhouette_score_data
    for ci_id, ci in cluster_result.items():
        silhouette_cluster = dict()
        for i in ci:
            a = 0
            for j in ci:
                if i != j:
                    a += distance(dataset_input, i, j)
            a = 0 if (len(ci) <= 1) else (a / (len(ci) - 1))

            cj_other_cluster = list()
            for cj_id, cj in cluster_result.items():
                if ci_id != cj_id:
                    b_temp = 0
                    for j in cj:
                        b_temp += distance(dataset_input, i, j)
                    b_temp = b_temp / len(cj)
                    cj_other_cluster.append(b_temp)
            b = min(cj_other_cluster)
            sil = 0 if (len(ci) <= 1) else (b - a) / max([a, b])
            avg += sil
            silhouette_cluster[i] = sil
            # print("Silhouette data", i, "Cluster ", ci_id, ":", sil)
        silhouette_score_data[ci_id] = sorted(
            silhouette_cluster.items(), key=lambda x: x[1])
        # print("-------------------------------------------------------")
    avg /= len(dataset_input)
    silhouette_score_data['avg'] = avg
    return silhouette_score_data


def average_silhouette(trained_weight, dataset_input):
    sil_data = silhouette(trained_weight, dataset_input)
    return sil_data['avg']


def silhouette_visualizer(trained_weight, dataset):
    silhouette_res = silhouette(trained_weight, dataset)
    import matplotlib.pyplot as plt
    import random as rd
    import base64
    from io import BytesIO

    img = BytesIO()

    fig, ax = plt.subplots(figsize=(6, 4), dpi=80)

    x = list()
    y = list()

    cmap = list()
    bcmap = list()
    count = 1
    for idj, cluster in silhouette_res.items():
        if idj != 'avg':
            r, g, b = (rd.random(), rd.random(), rd.random())
            random_color = (r, g, b, 1)
            random_bcolor = (r, g, b, 0.4)
            for i in cluster:
                x.append(i[1])
                y.append(count)
                cmap.append(random_color)
                bcmap.append(random_bcolor)
                count += 1

    ax.barh(y, x, color=cmap)
    ax.barh(y, [1] * len(dataset), color=bcmap)
    ax.get_yaxis().set_ticks([])
    ax.axvline(silhouette_res['avg'], ls='--', color='r')
    plt.text(silhouette_res['avg'], len(dataset) + 5, 'avg silhouette: ' + str(round(silhouette_res['avg'], 2)))
    title = 'Silhouette Result of ' + str(len(silhouette_res) - 1) + ' Cluster(s)\n\n'
    plt.title(title)
    plt.xlabel('Silhouette Score')
    plt.ylabel('Input dataset')

    plt.savefig(img, format='png')
    plt.close()

    img.seek(0)

    base64_bytes = base64.b64encode(img.getvalue())

    return base64_bytes.decode('utf-8')
     # Opt.: os.system("rm "+strFile)
