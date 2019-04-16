from application import routes
from main_algorithm import self_organizing_maps
import tests
from pandas import DataFrame as df


def test_clustering_visualization():
	manualisasi_df = df.from_csv("dataset_used/manualisasi.csv")
    manualisasi_df = manualisasi_df.drop(
        ["TUNA SUSILA", "ANAK BALITA TERLANTAR"], axis=1)
    
    hasil_cluster = routes.clustering_visualization(manu):
	pass