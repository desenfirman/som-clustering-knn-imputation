import self_organizing_maps


def test_normalize_data():
    return 0


def test_init_som_net():
    weight = self_organizing_maps.init_som_net(4, 5, 6)
    assert len(weight) == 4
    assert len(weight[0]) == 5
    assert len(weight[0][0]) == 6
