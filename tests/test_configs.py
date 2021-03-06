import itertools

import numpy as np
import pytest
from keras import Model

from main import main


def configs_fixture():
    model_configs = ['vlae', 'vanilla_vae', 'large_vanilla_vae', 'alexnet_classifier', 'simple_classifier',
                     'alexnet_vae', 'hvae']
    data_paths = ["./data/"]
    batch_sizes = [8, 16]
    z_dims = [["10", "11", "12"], ["13", "14", "15"]]
    use_dropouts = [True, False]
    use_batch_norms = [True, False]
    datasets = ['mnist']
    feature_map_reduction_factors = [1, 2]

    config_perms = itertools.product(model_configs, data_paths, batch_sizes, z_dims, use_dropouts, use_batch_norms,
                                     datasets,
                                     feature_map_reduction_factors)
    return list(config_perms)


def datasets_fixture():
    return [('mnist', 7, 7), ('celeba', 45, 16), ('imagenet', 6000, 0), ('cifar10', 3, 3)]


@pytest.mark.parametrize('config', configs_fixture())
def test_configs(mocker, config):
    mocker.patch.object(Model, 'fit')
    mocker.patch.object(Model, 'fit_generator')
    mocker.patch('keras.datasets.cifar10.load_data',
                 return_value=((np.ones((3, 32, 32, 3), dtype=np.uint8), np.ones(3, dtype=np.uint8)),
                               (np.ones((3, 32, 32, 3), dtype=np.uint8), np.ones(
                                   3, dtype=np.uint8))))
    mocker.patch('keras.datasets.mnist.load_data',
                 return_value=((np.ones((3, 28, 28, 3), dtype=np.uint8), np.ones(3, dtype=np.uint8)),
                               (np.ones((3, 28, 28, 3), dtype=np.uint8), np.ones(
                                   3, dtype=np.uint8))))
    cl_config = [
        "--num_epochs=1",
        "--steps_per_epoch=2",
        "--rgb=True",
        "--logdir=test",
        "--configuration={}".format(config[0]),
        "--data_path={}".format(config[1]),
        "--batch_size={}".format(config[2]),
        "--z_dim"
    ]
    if config[0] not in ['vlae', 'hvae']:
        cl_config.append(config[3][0])
    elif config[0] == 'vlae':
        cl_config += config[3]
    else:
        cl_config += config[3] + config[3][:-1]
    cl_config += ["--use_dropout={}".format(config[4]),
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
    mocker.patch('keras.datasets.cifar10.load_data',
                 return_value=((np.ones((3, 32, 32, 3), dtype=np.uint8), np.ones(3, dtype=np.uint8)),
                               (np.ones((3, 32, 32, 3), dtype=np.uint8), np.ones(
                                   3, dtype=np.uint8))))
    mocker.patch('keras.datasets.mnist.load_data',
                 return_value=((np.ones((7, 28, 28, 3), dtype=np.uint8), np.ones(7, dtype=np.uint8)),
                               (np.ones((7, 28, 28, 3), dtype=np.uint8), np.ones(
                                   7, dtype=np.uint8))))
    cl_config = [
        "--num_epochs=1",
        "--steps_per_epoch=2",
        "--configuration=vanilla_vae",
        "--data_path=./data/",
        "--logdir=test",
        "--batch_size=32",
        "--rgb=True",
        "--z_dims",
        "2",
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


@pytest.mark.parametrize('configuration',
                         ['vlae', 'vanilla_vae', 'large_vanilla_vae', 'alexnet_classifier', 'simple_classifier',
                          'alexnet_vae'])
@pytest.mark.parametrize('dataset', ['mnist', 'cifar10'])
def test_train_on_small_mnist(mocker, configuration, dataset):
    pass
    '''mocker.patch('keras.datasets.cifar10.load_data',
                 return_value=((np.ones((3, 32, 32, 3), dtype=np.uint8), np.ones(3, dtype=np.uint8)),
                               (np.ones((3, 32, 32, 3), dtype=np.uint8), np.ones(
                                   3, dtype=np.uint8))))
    mocker.patch('keras.datasets.mnist.load_data',
                 return_value=((np.ones((7, 28, 28, 3), dtype=np.uint8), np.ones(7, dtype=np.uint8)),
                               (np.ones((7, 28, 28, 3), dtype=np.uint8), np.ones(
                                   7, dtype=np.uint8))))
    cl_config = [
        "--num_epochs=1",
        "--steps_per_epoch=2",
        "--configuration={}".format(configuration),
        "--data_path=./data/",
        "--batch_size=32",
        "--logdir=test",
        "--rgb=True",
        "--z_dim=2",
        "--use_dropout=True",
        "--use_batch_norm=True",
        "--dataset={}".format(dataset)
    ]
    main(cl_config)'''


def test_frozen_alexnet():
    # TODO implement
    assert True


def test_alexnet_vae_classification_loss():
    # TODO implement
    assert True
