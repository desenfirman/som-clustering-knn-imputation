from main_algorithm import knn_imputation
from pandas import DataFrame as df
import tests


def test_knn_imputation():
    manualisasi_df = df.from_csv('dataset_used/manualisasi.csv')
    input_dataset = manualisasi_df.iloc[:, :].values
    imputed_dataset = knn_imputation.impute_dataset(input_dataset, 3)
    assert tests.is_almost_equal(imputed_dataset[0][0], 12.33333, 5)
    assert tests.is_almost_equal(imputed_dataset[0][5], 6701, 5)
    assert tests.is_almost_equal(imputed_dataset[9][5], 1108, 5)
    assert tests.is_almost_equal(imputed_dataset[9][0], 188.333, 5)
    assert tests.is_almost_equal(imputed_dataset[3][2], 32, 5)


def test_no_knn_imputation():
    manualisasi_df = df.from_csv('dataset_used/manualisasi.csv')
    input_dataset = manualisasi_df.iloc[:, :].values
    imputed_dataset = knn_imputation.impute_dataset(input_dataset, 0)
    assert tests.is_almost_equal(imputed_dataset[0][0], 0, 5)
    assert tests.is_almost_equal(imputed_dataset[0][5], 0, 5)
    assert tests.is_almost_equal(imputed_dataset[9][0], 0, 5)


def test_get_dist_pair():
    manualisasi_df = df.from_csv('dataset_used/manualisasi.csv')
    input_dataset = manualisasi_df.iloc[:, :].values
    ex_dist_1 = knn_imputation.get_dist_pair(
        input_dataset)
    assert tests.is_almost_equal(len(ex_dist_1), 45, 5)
