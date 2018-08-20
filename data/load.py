import os

from torchvision import transforms, datasets
from data.stanford_dogs import StanDogs
from data.oxford_flowers import OxFlowers
import configs

def load_datasets(set_name, input_size=224):
    if set_name == 'mnist':
        train_dataset = datasets.MNIST(root=os.path.join(configs.general.paths.imagesets, 'MNIST'),
                                                   train=True,
                                                   transform=transforms.ToTensor(),
                                                   download=True)
        test_dataset = datasets.MNIST(root=os.path.join(configs.general.paths.imagesets, 'MNIST'),
                                                  train=False,
                                                  transform=transforms.ToTensor())

    elif set_name == 'stanford_dogs':
        input_transforms = transforms.Compose([
            transforms.RandomResizedCrop(input_size, ratio=(1, 1.3)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()])

        train_dataset = StanDogs(root=configs.general.paths.imagesets,
                                 train=True,
                                 cropped=False,
                                 transform=input_transforms,
                                 download=True)
        test_dataset = StanDogs(root=configs.general.paths.imagesets,
                                train=False,
                                cropped=False,
                                transform=input_transforms,
                                download=True)

        print("Training set stats:")
        train_dataset.stats()
        print("Testing set stats:")
        test_dataset.stats()

    elif set_name == 'oxford_flowers':
        input_transforms = transforms.Compose([
            transforms.RandomResizedCrop(input_size, ratio=(1, 1.3)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()])

        train_dataset = OxFlowers(root=configs.general.paths.imagesets,
                                  train=True,
                                  val=False,
                                  transform=input_transforms,
                                  download=True)
        test_dataset = OxFlowers(root=configs.general.paths.imagesets,
                                train=False,
                                val=True,
                                transform=input_transforms,
                                download=True)

        print("Training set stats:")
        train_dataset.stats()
        print("Testing set stats:")
        test_dataset.stats()
    else:
        return None, None

    return train_dataset, test_dataset