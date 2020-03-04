import itertools

import pytest

from main import main


def configs():
    model_configs = ['vanilla_vae', 'large_vanilla_vae', 'alexnet_classifier', 'simple_classifier', 'alexnet_vae',
                     'frozen_alexnet_vae', 'alexnet_vae_classification_loss']
    data_paths = ["../data/"]
    batch_sizes = [8, 32]
    z_dims = [10, 20]
    use_dropouts = [True, False]
    use_batch_norms = [True, False]
    datasets = ['mnist', 'celeba', 'imagenet', 'cifar10']
    feature_map_reduction_factors = [1, 2]

    config_perms = itertools.product(model_configs, data_paths, batch_sizes, z_dims, use_dropouts, use_batch_norms,
                                     datasets,
                                     feature_map_reduction_factors)
    return list(config_perms)


@pytest.mark.parametrize('config', configs(), ids=[" ".join(map(lambda x: str(x), c)) for c in configs()])
def test_configs(config):
    cl_config = [
        "--num_epochs=1",
        "--configuration={}".format(config[0]),
        "--feature_map_layers", "1",
        "--data_path={}".format(config[1]),
        "--batch_size={}".format(config[2]),
        "--z_dim={}".format(config[3]),
        "--use_dropout={}".format(config[4]),
        "--use_batch_norm={}".format(config[5]),
        "--dataset={}".format(config[6]),
        "--feature_map_reduction_factor={}".format(config[7])
    ]
    print("run config {}".format(cl_config))
    main(cl_config)
