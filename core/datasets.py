import cv2
import numpy as np
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms


def get_unsupervised_dataset(config):
    if hasattr(datasets, config['dataset']):
        if 'CIFAR' in config['dataset']:
            transform = cifar_transforms(config['s'], config['input_shape'])
            train_set = getattr(datasets, config['dataset'])
            train_set = train_set(root=config['root'], train=True, download=True)
        elif 'ImageFolder' == config['dataset']:
            transform = imagenet_transforms(config['s'], config['input_shape'])
            train_set = datasets.ImageFolder(root=config['root'])
    elif 'ImageNet' == config['dataset']:
        transform = imagenet_transforms(config['s'], config['input_shape'])
        train_set = datasets.ImageFolder(root=config['root'])
    else:
        raise ValueError('Dataset {} is not implemented'.format(config['dataset']))

    dataset = Unsupervised_Dataset(train_set, transform)
    dataloader = data.DataLoader(dataset, batch_size=config['batch_size'],
                                 shuffle=True, num_workers=config['num_workers'],
                                 drop_last=True)
    return dataloader


def get_supervised_dataset(config):
    if hasattr(datasets, config['dataset']):
        if 'CIFAR' in config['dataset']:
            tt, vt = cifar_linear_transforms()
            loader = getattr(datasets, config['dataset'])
            train_set = loader(root=config['root'], train=True, download=True,
                                  transform=tt)
            val_set = loader(root=config['val_root'], train=False, download=True,
                                transform=vt)
        elif 'ImageFolder' == config['dataset']:
            tt, vt = imagenet_linear_transform()
            train_set = datasets.ImageFolder(root=config['root'],
                                             transform=tt)
            val_set = datasets.ImageFolder(root=config['val_root'],
                                           transform=vt)
    elif 'ImageNet' == config['dataset']:
        tt, vt = imagenet_linear_transform()
        train_set = datasets.ImageFolder(root=config['root'],
                                         transform=tt)
        val_set = datasets.ImageFolder(root=config['val_root'],
                                       transform=vt)
    else:
        raise ValueError('Dataset {} is not implemented'.format(config['dataset']))

    
    train_loader = data.DataLoader(train_set, batch_size=config['batch_size'],
                                   shuffle=True, num_workers=config['num_workers'],
                                   drop_last=False)
    val_loader = data.DataLoader(val_set, batch_size=config['batch_size'],
                                 shuffle=False, num_workers=config['num_workers'],
                                 drop_last=False)
    return train_loader, val_loader


#####################
### DATASET UTILS ###
#####################


class Unsupervised_Dataset():
    r'''
    Takes a dataset as inputs
    :param dataset: dataset pytorch-style where __getitem__(idx) return an image and label
                    the dataset must NOT contain any transform
    '''
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, _ = self.dataset[idx]

        img1 = self.transform(img)
        img2 = self.transform(img)

        return img1, img2


def imagenet_transforms(s, input_shape):
    # get a set of data augmentation transformations as described in the SimCLR paper.
    color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=input_shape[0]),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.RandomApply([color_jitter], p=0.8),
                                          transforms.RandomGrayscale(p=0.2),
                                          GaussianBlur(kernel_size=int(0.1 *input_shape[0])),
                                          transforms.ToTensor()])
    return data_transforms


def cifar_transforms(s, input_shape):
    # get a set of data augmentation transformations as described in the SimCLR paper.
    color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=input_shape[0]),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.RandomApply([color_jitter], p=0.8),
                                          transforms.RandomGrayscale(p=0.2),
                                          transforms.ToTensor()])
    return data_transforms

def cifar_linear_transforms():
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    transform_test = transforms.ToTensor()

    return transform_train, transform_test


def imagenet_linear_transform():
    tt = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
    ])

    vt = transforms.Compose([transforms.Resize(256),
                             transforms.CenterCrop(224),
                             transforms.ToTensor()])

    return tt, vt
    
            


class GaussianBlur():
    # Implements Gaussian blur as described in the SimCLR paper
    def __init__(self, kernel_size, min=0.1, max=2.0):
        self.min = min
        self.max = max
        # kernel size is set to be 10% of the image height/width
        self.kernel_size = kernel_size

    def __call__(self, sample):
        sample = np.array(sample)
        if np.random.random_sample() < 0.5:
            sigma = (self.max - self.min) * np.random.random_sample() + self.min
            sample = cv2.GaussianBlur(sample, (self.kernel_size, self.kernel_size), sigma)

        return sample