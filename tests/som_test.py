from main_algorithm import self_organizing_maps
import tests
from pandas import DataFrame as df


# test case function
def test_normalize_data():
    manualisasi_df = df.from_csv("dataset_used/manualisasi.csv")
    manualisasi_df = manualisasi_df.drop(
        ["TUNA SUSILA", "ANAK BALITA TERLANTAR"], axis=1)
    input_dataset = manualisasi_df.iloc[:, :].values
    normalized_data = self_organizing_maps.normalize_data(input_dataset)
    assert tests.is_almost_equal(normalized_data[0][2], 0.09961686, 5)


def test_init_som_net():
    weight = self_organizing_maps.init_som_net(4, 5, 6)
    assert len(weight) == 4
    assert len(weight[0]) == 5
    assert len(weight[0][0]) == 6
    assert weight[0][0][0] > 0.0 and weight[0][0][0] < 1.0
    assert weight[3][4][5] > 0.0 and weight[3][4][5] < 1.0


def test_training():
    dataset_input = [
        [0.0310458, 0.0996169, 0.336344],
        [0.8088235, 0.9923372, 0.1495946],
        [0.8937908, 1, 0.0493175]
    ]
    weight = [
        [
            [0.8940559, 0.7651827, 0.2653192],  # neuron 0,0
            [0.1737489, 0.4458108, 0.1619524]   # neuron 0,1
        ],
        [
            [0.4348796, 0.9573226, 0.4023138],  # neuron 1,0
            [0.2289554, 0.0225801, 0.1493896]   # neuron 1,1
        ]]
    # som net testing
    assert len(weight) == 2
    assert len(weight[0]) == 2
    assert len(weight[0][0]) == 3

    trained_weight = self_organizing_maps.training(
        dataset_input, weight, 3, 0.5, 0.5)

    # som weight testing
    assert tests.is_almost_equal(trained_weight[0][0][0], 0.86263723, 5)
    assert tests.is_almost_equal(trained_weight[0][1][1], 0.45358939, 5)
    assert tests.is_almost_equal(trained_weight[1][0][2], 0.39544198, 5)
    assert tests.is_almost_equal(trained_weight[1][1][1], 0.07578458, 5)
    assert tests.is_almost_equal(trained_weight[0][1][2], 0.16242615, 5)
    assert tests.is_almost_equal(trained_weight[0][0][1], 0.97424479, 5)


def test_clustering():
    trained_weight = [
        [
            [0.86263723, 0.97424479, 0.1071898],  # neuron 0,0
            [0.18618813, 0.45358939, 0.16242615]   # neuron 0,1
        ],
        [
            [0.43933853, 0.94946919, 0.39544198],  # neuron 1,0
            [0.09307897, 0.07578458, 0.27788371]   # neuron 1,1
        ]]
    input_data_1 = [0.0310458, 0.0996169, 0.336344]
    test_1 = self_organizing_maps.penentuan_cluster(
        trained_weight, input_data_1)
    assert test_1[0] == 1 and test_1[1] == 1


def test_quantization_error():
    dataset_input = [
        [0.0310458, 0.0996169, 0.336344],
        [0.8088235, 0.9923372, 0.1495946],
        [0.8937908, 1, 0.0493175]
    ]
    trained_weight = [
        [
            [0.86263723, 0.97424479, 0.1071898],  # neuron 0,0
            [0.18618813, 0.45358939, 0.16242615]   # neuron 0,1
        ],
        [
            [0.43933853, 0.94946919, 0.39544198],  # neuron 1,0
            [0.09307897, 0.07578458, 0.27788371]   # neuron 1,1
        ]]
    qe_score = self_organizing_maps.quantization_error(
        trained_weight, dataset_input)
    assert tests.is_almost_equal(qe_score, 0.0766537, 5)


def test_davies_bouldin_index():
    dataset_input = [
        [0.0310458, 0.0996169, 0.336344],
        [0.8088235, 0.9923372, 0.1495946],
        [0.8937908, 1, 0.0493175]
    ]
    trained_weight = [
        [
            [0.86263723, 0.97424479, 0.1071898],  # neuron 0,0
            [0.18618813, 0.45358939, 0.16242615]   # neuron 0,1
        ],
        [
            [0.43933853, 0.94946919, 0.39544198],  # neuron 1,0
            [0.09307897, 0.07578458, 0.27788371]   # neuron 1,1
        ]]
    dbi_score = self_organizing_maps.davies_bouldin_index(
        trained_weight, dataset_input)
    assert tests.is_almost_equal(dbi_score, 0.0728676497137748, 5)
