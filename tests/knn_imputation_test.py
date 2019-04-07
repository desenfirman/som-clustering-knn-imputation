from main_algorithm import knn_imputation
from pandas import DataFrame as df
import tests


def test_knn_imputation():
    manualisasi_df = df.from_csv('dataset_used/manualisasi.csv')
    input_dataset = manualisasi_df.iloc[:, :].values
    imputed_dataset = knn_imputation.impute_dataset(input_dataset, 3)
    assert tests.is_almost_equal(imputed_dataset[0][0], 12.33333, 5)
    assert tests.is_almost_equal(imputed_dataset[0][5], 6701, 5)
    assert tests.is_almost_equal(imputed_dataset[0][2], 6, 5)
    assert tests.is_almost_equal(imputed_dataset[9][5], 1108, 5)


def test_get_sorted_distance_group():
    manualisasi_df = df.from_csv('dataset_used/manualisasi.csv')
    input_dataset = manualisasi_df.iloc[:, :].values
    ex_data = (0, 0)
    ex_dist_1 = knn_imputation.get_sorted_distance_group(
        input_dataset, ex_data)
    assert tests.is_almost_equal(ex_dist_1[0][2], 2.00291292625085, 5)

    ex_data_2 = (1, 1)
    ex_dist_2 = knn_imputation.get_sorted_distance_group(
        input_dataset, ex_data_2)
    assert tests.is_almost_equal(ex_dist_2[0][2], 1.11810708991569, 5)
