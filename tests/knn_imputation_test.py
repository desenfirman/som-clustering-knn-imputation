from main_algorithm import knn_imputation
from pandas import DataFrame as df
import tests


def test_knn_imputation():
    manualisasi_df = df.from_csv('dataset_used/manualisasi.csv')
    input_dataset = manualisasi_df.iloc[:, :].values
    imputed_dataset = knn_imputation.impute_dataset(input_dataset)
    assert tests.is_almost_equal(imputed_dataset[0][0], 12.33333, 5)
    assert tests.is_almost_equal(imputed_dataset[0][5], 6701, 5)


def test_get_sorted_distance_group():
    input_dataset = list()
    ex_dist_1 = knn_imputation.get_sorted_distance_group(input_dataset, 0)
    assert tests.is_almost_equal(ex_dist_1[0][1], 0.107982360331431, 5)
