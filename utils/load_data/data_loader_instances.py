import os
import pickle
from torchvision import datasets
import numpy as np
from scipy.io import loadmat
from .base_load_data import base_load_data
import wget

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

# ======================================================================================================================
# Dataset-specific loader classes
# Each class inherits base_load_data and overrides obtain_data() (and optionally
# seperate_data_from_label / preprocessing_ / load_dataset) for its own format.
# ======================================================================================================================

class dynamic_mnist_loader(base_load_data):
    def __init__(self, args, use_fixed_validation=False, no_binarization=False):
        super(dynamic_mnist_loader, self).__init__(args, use_fixed_validation, no_binarization=no_binarization)

    def obtain_data(self):
        # download MNIST on first run; reuse cached copy afterwards
        train = datasets.MNIST(os.path.join('datasets', self.args.dataset_name), train=True, download=True)
        test = datasets.MNIST(os.path.join('datasets', self.args.dataset_name), train=False)
        return train, test


class fashion_mnist_loader(base_load_data):
    def __init__(self, args, use_fixed_validation=False, no_binarization=False):
        super(fashion_mnist_loader, self).__init__(args, use_fixed_validation, no_binarization=no_binarization)

    def obtain_data(self):
        # download FashionMNIST on first run; reuse cached copy afterwards
        train = datasets.FashionMNIST(os.path.join('datasets', self.args.dataset_name), train=True, download=True)
        test = datasets.FashionMNIST(os.path.join('datasets', self.args.dataset_name), train=False)
        return train, test


class svhn_loader(base_load_data):
    def __init__(self, args, use_fixed_validation=False, no_binarization=False):
        super(svhn_loader, self).__init__(args, use_fixed_validation, no_binarization=no_binarization)

    def obtain_data(self):
        # SVHN uses split='train'/'test' instead of train=True/False
        train = datasets.SVHN(os.path.join('datasets', self.args.dataset_name), split='train', download=True)
        test = datasets.SVHN(os.path.join('datasets', self.args.dataset_name), split='test', download=True)
        return train, test

    def seperate_data_from_label(self, train_dataset, test_dataset):
        # SVHN stores labels in .labels (not .train_labels / .test_labels like MNIST)
        x_train = train_dataset.data
        y_train = train_dataset.labels.astype(dtype=int)
        x_test = test_dataset.data
        y_test = test_dataset.labels.astype(dtype=int)
        return x_train, y_train, x_test, y_test


# ======================================================================================================================
# Static MNIST — pre-binarized train/val/test shipped as .amat text files
# No dynamic binarization; val set is pre-built (not carved out of train).
# ======================================================================================================================
class static_mnist_loader(base_load_data):
    def __init__(self, args, use_fixed_validation=False, no_binarization=False):
        super(static_mnist_loader, self).__init__(args, use_fixed_validation, no_binarization=no_binarization)

    def obtain_data(self):
        def lines_to_np_array(lines):
            return np.array([[int(i) for i in line.split()] for line in lines])

        # read the three pre-split .amat files (space-separated 0/1 pixels)
        with open(os.path.join('datasets', self.args.dataset_name, 'binarized_mnist_train.amat')) as f:
            lines = f.readlines()
        x_train = lines_to_np_array(lines).astype('float32')
        with open(os.path.join('datasets', self.args.dataset_name, 'binarized_mnist_valid.amat')) as f:
            lines = f.readlines()
        x_val = lines_to_np_array(lines).astype('float32')
        with open(os.path.join('datasets', self.args.dataset_name, 'binarized_mnist_test.amat')) as f:
            lines = f.readlines()
        x_test = lines_to_np_array(lines).astype('float32')

        # static MNIST has no class labels — fill with zeros so the rest of the pipeline stays uniform
        y_train = np.zeros((x_train.shape[0], 1)).astype(int)
        y_val = np.zeros((x_val.shape[0], 1)).astype(int)
        y_test = np.zeros((x_test.shape[0], 1)).astype(int)
        # pack val into the train tuple so base_load_data.load_dataset() can unpack it
        return (x_train, x_val, y_train, y_val), (x_test, y_test)

    def seperate_data_from_label(self, train_dataset, test_dataset):
        # unpack the custom tuple format returned by obtain_data above
        x_train, x_val, y_train, y_val = train_dataset
        x_test, y_test = test_dataset
        # return (x_train, x_val) and (y_train, y_val) so load_dataset() can split them at the static_mnist branch
        return (x_train, x_val), (y_train, y_val), x_test, y_test

    def preprocessing_(self, x_train, x_test):
        # data is already binarized — skip pixel normalization
        return x_train, x_test


# ======================================================================================================================
# Omniglot — handwritten characters dataset loaded from a .mat file
# Pixels are stored in Fortran (column-major) order and must be transposed.
# ======================================================================================================================
class omniglot_loader(base_load_data):
    def __init__(self, args, use_fixed_validation=False, no_binarization=False):
        super(omniglot_loader, self).__init__(args, use_fixed_validation, no_binarization=no_binarization)

    def obtain_data(self):
        def reshape_data(data):
            # data is stored flattened in Fortran (column-major) order — reshape accordingly
            return data.reshape((-1, 28, 28)).reshape((-1, 28*28), order='F')

        dataset_file = os.path.join('datasets', self.args.dataset_name, 'chardata.mat')
        if not os.path.exists(dataset_file):
            url = "https://raw.githubusercontent.com/yburda/iwae/master/datasets/OMNIGLOT/chardata.mat"
            wget.download(url, dataset_file)

        omni_raw = loadmat(os.path.join('datasets', self.args.dataset_name, 'chardata.mat'))

        x_train = reshape_data(omni_raw['data'].T.astype('float32'))
        x_test = reshape_data(omni_raw['testdata'].T.astype('float32'))

        y_train = omni_raw['targetchar'].reshape((-1, 1))
        y_test = omni_raw['testtargetchar'].reshape((-1, 1))
        return (x_train, y_train), (x_test, y_test)

    def seperate_data_from_label(self, train_dataset, test_dataset):
        # data and labels are packed as tuples in obtain_data
        x_train, y_train = train_dataset
        x_test, y_test = test_dataset
        return x_train, y_train, x_test, y_test

    def preprocessing_(self, x_train, x_test):
        # data is already in [0, 1] from the .mat file — skip pixel normalization
        return x_train, x_test


# ======================================================================================================================
# CIFAR-10 — 32x32 RGB images; axes are swapped to (C, H, W) order expected by the model
# ======================================================================================================================
class cifar10_loader(base_load_data):
    def __init__(self, args, use_fixed_validation=False, no_binarization=False):
        super(cifar10_loader, self).__init__(args, use_fixed_validation, no_binarization=no_binarization)

    def obtain_data(self):
        training_dataset = datasets.CIFAR10(os.path.join('datasets', self.args.dataset_name), train=True, download=True)
        test_dataset = datasets.CIFAR10(os.path.join('datasets', self.args.dataset_name), train=False)
        return training_dataset, test_dataset

    def seperate_data_from_label(self, train_dataset, test_dataset):
        # CIFAR-10 stores images as (N, H, W, C); swap to (N, C, H, W)
        train_data = np.swapaxes(np.swapaxes(train_dataset.data, 1, 2), 1, 3)
        y_train = np.zeros((train_data.shape[0], 1)).astype(int)
        test_data = np.swapaxes(np.swapaxes(test_dataset.data, 1, 2), 1, 3)
        y_test = np.zeros((test_data.shape[0], 1)).astype(int)
        return train_data, y_train, test_data, y_test


# ======================================================================================================================
# Caltech 101 Silhouettes — binary 28x28 images from a .mat file with a pre-built train/val/test split.
# Overrides load_dataset() directly because the base class split logic does not apply.
# Pseudo-inputs default: mean=0.5, std=0.02
# ======================================================================================================================
class caltech101silhouettes_loader(base_load_data):
    def __init__(self, args):
        # caltech has its own train/val/test split so we skip the base class splitting
        super(caltech101silhouettes_loader, self).__init__(args)

    def obtain_data(self):
        pass  # not used — load_dataset is overridden

    def load_dataset(self, **kwargs):
        # start processing
        def reshape_data(data):
            # silhouettes are stored flattened in Fortran order — reshape accordingly
            return data.reshape((-1, 28, 28)).reshape((-1, 28 * 28), order='fortran')

        caltech_raw = loadmat(os.path.join('datasets', 'Caltech101Silhouettes',
                                           'caltech101_silhouettes_28_split1.mat'))

        # invert: original 1=background, 0=foreground — flip so 1=foreground
        x_train = 1. - reshape_data(caltech_raw['train_data'].astype('float32'))
        np.random.shuffle(x_train)
        x_val = 1. - reshape_data(caltech_raw['val_data'].astype('float32'))
        np.random.shuffle(x_val)
        x_test = 1. - reshape_data(caltech_raw['test_data'].astype('float32'))

        y_train = caltech_raw['train_labels']
        y_val = caltech_raw['val_labels']
        y_test = caltech_raw['test_labels']

        train_loader, val_loader, test_loader = self.post_processing(
            x_train, x_val, x_test, y_train, y_val, y_test,
            init_mean=0.5, init_std=0.02, **kwargs)

        return train_loader, val_loader, test_loader, self.args


# ======================================================================================================================
# HistopathologyGray — grayscale 28x28 patches from a pickle file with a pre-built split.
# Values clipped to [1/512, 1-1/512] to avoid log(0) in continuous likelihood.
# Overrides load_dataset() directly because the base class split logic does not apply.
# Pseudo-inputs default: mean=0.4, std=0.05
# ======================================================================================================================
class histopathologyGray_loader(base_load_data):
    def __init__(self, args):
        # histopathology has its own train/val/test split
        super(histopathologyGray_loader, self).__init__(args)

    def obtain_data(self):
        pass  # not used — load_dataset is overridden

    def load_dataset(self, **kwargs):
        # start processing
        with open(os.path.join('datasets', 'HistopathologyGray', 'histopathology.pkl'), 'rb') as f:
            data = pickle.load(f)

        # pre-split in pickle file
        x_train = np.asarray(data['training']).reshape(-1, 28 * 28)
        x_val = np.asarray(data['validation']).reshape(-1, 28 * 28)
        x_test = np.asarray(data['test']).reshape(-1, 28 * 28)

        # clip to avoid log(0) in likelihood computation
        x_train = np.clip(x_train, 1. / 512., 1. - 1. / 512.)
        x_val = np.clip(x_val, 1. / 512., 1. - 1. / 512.)
        x_test = np.clip(x_test, 1. / 512., 1. - 1. / 512.)

        # idle labels
        y_train = np.zeros((x_train.shape[0], 1))
        y_val = np.zeros((x_val.shape[0], 1))
        y_test = np.zeros((x_test.shape[0], 1))

        train_loader, val_loader, test_loader = self.post_processing(
            x_train, x_val, x_test, y_train, y_val, y_test,
            init_mean=0.4, init_std=0.05, **kwargs)

        return train_loader, val_loader, test_loader, self.args


# ======================================================================================================================
# FreyFaces — grayscale 28x20 face images loaded from a pickle file.
# Overrides load_dataset() directly because the base class split logic does not apply.
# Pseudo-inputs default: mean=0.5, std=0.02
# ======================================================================================================================
class freyfaces_loader(base_load_data):
    TRAIN = 1565
    VAL = 200
    TEST = 200

    def __init__(self, args):
        super(freyfaces_loader, self).__init__(args)

    def obtain_data(self):
        pass  # not used — load_dataset is overridden

    def load_dataset(self, **kwargs):
        # start processing
        with open(os.path.join('datasets', 'Freyfaces', 'freyfaces.pkl'), 'rb') as f:
            data = pickle.load(f)

        # preprocessing: shift by 0.5 and scale to [0, 1]
        data = (data[0] + 0.5) / 256.

        # shuffle data
        np.random.shuffle(data)

        # split into train / validation / test
        x_train = data[0:self.TRAIN].reshape(-1, 28 * 20)
        x_val = data[self.TRAIN:(self.TRAIN + self.VAL)].reshape(-1, 28 * 20)
        x_test = data[(self.TRAIN + self.VAL):(self.TRAIN + self.VAL + self.TEST)].reshape(-1, 28 * 20)

        # idle labels
        y_train = np.zeros((x_train.shape[0], 1))
        y_val = np.zeros((x_val.shape[0], 1))
        y_test = np.zeros((x_test.shape[0], 1))

        train_loader, val_loader, test_loader = self.post_processing(
            x_train, x_val, x_test, y_train, y_val, y_test,
            init_mean=0.5, init_std=0.02, **kwargs)

        return train_loader, val_loader, test_loader, self.args


# ======================================================================================================================
# Top-level dispatch function
# Auto-configures runtime args (input_size, input_type, dynamic_binarization, training_set_size)
# for each dataset and instantiates the correct loader.
# ======================================================================================================================
def load_dataset(args, training_num=None, use_fixed_validation=False, no_binarization=False, **kwargs):
    # runtime field: training_set_size can be overridden by caller
    if training_num is not None:
        args.training_set_size = training_num
    if args.dataset_name == 'static_mnist':
        # runtime fields: overrides CLI defaults based on dataset
        args.input_size = [1, 28, 28]
        args.input_type = 'binary'
        train_loader, val_loader, test_loader, args = static_mnist_loader(args).load_dataset(**kwargs)
    elif args.dataset_name == 'dynamic_mnist':
        # runtime fields: overrides CLI defaults based on dataset
        if training_num is None:
            args.training_set_size = 50000
        args.input_size = [1, 28, 28]
        if args.continuous is True:
            # continuous mode: treat as grayscale, skip binarization
            args.input_type = 'gray'
            args.dynamic_binarization = False
            no_binarization = True
        else:
            args.input_type = 'binary'
            args.dynamic_binarization = True

        train_loader, val_loader, test_loader, args = \
            dynamic_mnist_loader(args, use_fixed_validation, no_binarization=no_binarization).load_dataset(**kwargs)
    elif args.dataset_name == 'fashion_mnist':
        # runtime fields: overrides CLI defaults based on dataset
        if training_num is None:
            args.training_set_size = 50000
        args.input_size = [1, 28, 28]

        if args.continuous is True:
            print("*****Continuous Data*****")
            # continuous mode: treat as grayscale, skip binarization
            args.input_type = 'gray'
            args.dynamic_binarization = False
            no_binarization = True
        else:
            args.input_type = 'binary'
            args.dynamic_binarization = True

        train_loader, val_loader, test_loader, args = \
            fashion_mnist_loader(args, use_fixed_validation, no_binarization=no_binarization).load_dataset(**kwargs)
    elif args.dataset_name == 'omniglot':
        # runtime fields: overrides CLI defaults based on dataset
        if training_num is None:
            args.training_set_size = 23000
        args.input_size = [1, 28, 28]
        args.input_type = 'binary'
        args.dynamic_binarization = True
        train_loader, val_loader, test_loader, args = omniglot_loader(args).load_dataset(**kwargs)
    elif args.dataset_name == 'svhn':
        # runtime fields: overrides CLI defaults based on dataset
        args.training_set_size = 60000
        args.input_size = [3, 32, 32]
        args.input_type = 'continuous'
        train_loader, val_loader, test_loader, args = svhn_loader(args).load_dataset(**kwargs)
    elif args.dataset_name == 'cifar10':
        # runtime fields: overrides CLI defaults based on dataset
        args.training_set_size = 40000
        args.input_size = [3, 32, 32]
        args.input_type = 'continuous'
        train_loader, val_loader, test_loader, args = cifar10_loader(args).load_dataset(**kwargs)
    elif args.dataset_name == 'caltech101silhouettes':
        # runtime fields: overrides CLI defaults based on dataset
        args.input_size = [1, 28, 28]
        args.input_type = 'binary'
        args.dynamic_binarization = False
        train_loader, val_loader, test_loader, args = caltech101silhouettes_loader(args).load_dataset(**kwargs)
    elif args.dataset_name == 'histopathologyGray':
        # runtime fields: overrides CLI defaults based on dataset
        args.input_size = [1, 28, 28]
        args.input_type = 'gray'
        args.dynamic_binarization = False
        train_loader, val_loader, test_loader, args = histopathologyGray_loader(args).load_dataset(**kwargs)
    elif args.dataset_name == 'freyfaces':
        # runtime fields: overrides CLI defaults based on dataset
        args.input_size = [1, 28, 20]
        args.input_type = 'gray'
        args.dynamic_binarization = False
        train_loader, val_loader, test_loader, args = freyfaces_loader(args).load_dataset(**kwargs)
    elif args.dataset_name == 'ecg5000':
        from .timeseries_loader import ecg5000_loader
        args.training_set_size = 4000
        train_loader, val_loader, test_loader, args = ecg5000_loader(args).load_dataset(**kwargs)
    elif args.dataset_name == 'synthetic_timeseries':
        from .timeseries_loader import synthetic_timeseries_loader
        args.training_set_size = 4000
        train_loader, val_loader, test_loader, args = synthetic_timeseries_loader(args).load_dataset(**kwargs)
    elif args.dataset_name == 'csv_timeseries':
        from .timeseries_loader import csv_timeseries_loader
        args.training_set_size = 0  # placeholder; overridden inside load_dataset()
        train_loader, val_loader, test_loader, args = csv_timeseries_loader(args).load_dataset(**kwargs)
    elif args.dataset_name == 'parquet_timeseries':
        from .timeseries_loader import parquet_timeseries_loader
        args.training_set_size = 0
        train_loader, val_loader, test_loader, args = parquet_timeseries_loader(args).load_dataset(**kwargs)
    else:
        raise Exception('Wrong name of the dataset!')
    print('train size', len(train_loader.dataset))
    if val_loader is not None:
        print('val size', len(val_loader.dataset))
    print('test size', len(test_loader.dataset))
    return train_loader, val_loader, test_loader, args
