import itertools

import pytest
from keras import Model

from main import main


def configs_fixture():
    model_configs = ['vanilla_vae', 'large_vanilla_vae', 'alexnet_classifier', 'simple_classifier', 'alexnet_vae',
                     'alexnet_vae_classification_loss']
    data_paths = ["../data/"]
    batch_sizes = [8]
    z_dims = [10]
    use_dropouts = [True, False]
    use_batch_norms = [True, False]
    datasets = ['mnist']
    feature_map_reduction_factors = [1, 2]

    config_perms = itertools.product(model_configs, data_paths, batch_sizes, z_dims, use_dropouts, use_batch_norms,
                                     datasets,
                                     feature_map_reduction_factors)
    return list(config_perms)


def datasets_fixture():
    return [('mnist', 60000, 10000), ('celeba', 144, 16), ('imagenet', 6000, 0), ('cifar10', 50000, 10000)]


@pytest.mark.parametrize('config', configs_fixture())
def test_configs(mocker, config):
    mocker.patch.object(Model, 'fit')
    mocker.patch.object(Model, 'fit_generator')
    cl_config = [
        "--num_epochs=1",
        "--steps_per_epoch=2",
        "--configuration={}".format(config[0]),
        "--data_path={}".format(config[1]),
        "--batch_size={}".format(config[2]),
        "--z_dim={}".format(config[3]),
        "--use_dropout={}".format(config[4]),
        "--use_batch_norm={}".format(config[5]),
        "--dataset={}".format(config[6]),
        "--feature_map_reduction_factor={}".format(config[7])
    ]
    main(cl_config)
    assert Model.fit.called or Model.fit_generator.called


@pytest.mark.parametrize('dataset', datasets_fixture())
def test_datasets_pass(mocker, dataset):
    mocker.patch.object(Model, 'fit')
    mocker.patch.object(Model, 'fit_generator')
    cl_config = [
        "--num_epochs=1",
        "--steps_per_epoch=2",
        "--configuration=vanilla_vae",
        "--data_path=../data/",
        "--batch_size=32",
        "--z_dim=2",
        "--use_dropout=True",
        "--use_batch_norm=True",
        "--dataset={}".format(dataset[0])
    ]
    main(cl_config)
    assert Model.fit.called or Model.fit_generator.called
    ds = Model.fit.call_args[1] if Model.fit.called else Model.fit_generator.call_args[1]

    if 'x' in ds:
        assert len(ds['x']) == dataset[1]
    else:
        assert ds['generator'].n == dataset[1]
