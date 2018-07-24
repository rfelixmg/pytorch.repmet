from __future__ import print_function
from PIL import Image
from os.path import join
import os
import torch.utils.data as data
from torchvision.datasets.utils import download_url, list_dir, list_files


class StanDogs(data.Dataset):
    """`Stanford Dogs <http://vision.stanford.edu/aditya86/ImageNetDogs/>`_ Dataset.
    Args:
        root (string): Root directory of dataset where directory
            ``omniglot-py`` exists.
        cropped (bool, optional): If true, the images will be cropped into the bounding box specified
            in the annotations
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset tar files from the internet and
            puts it in root directory. If the tar files are already downloaded, they are not
            downloaded again.
    """
    folder = 'StanfordDogs'
    download_url_prefix = 'http://vision.stanford.edu/aditya86/ImageNetDogs'

    def __init__(self,
                 root,
                 cropped=False,
                 transform=None,
                 target_transform=None,
                 download=False):

        self.root = join(os.path.expanduser(root), self.folder)
        self.cropped = cropped
        self.transform = transform
        self.target_transform = target_transform

        if download:
            self.download()

        self.images_folder = join(self.root, 'Images')
        self.annotations_folder = join(self.root, 'Annotation')
        self._breeds = list_dir(self.images_folder)

        if self.cropped:
            self._breed_annotations = [[[(annotation, box, idx)
                                        for box in self.get_boxes(join(self.annotations_folder, breed, annotation))]
                                        for annotation in list_files(join(self.annotations_folder, breed), '')]
                                        for idx, breed in enumerate(self._breeds)]
            self._flat_breed_annotations = sum(sum(self._breed_annotations, []), [])

            self._flat_breed_images = [(annotation+'.jpg', idx) for annotation, box, idx in self._flat_breed_annotations]
        else:
            self._breed_images = [[(image, idx) for image in list_files(join(self.images_folder, breed), '.jpg')]
                                      for idx, breed in enumerate(self._breeds)]

            self._flat_breed_images = sum(self._breed_images, [])

        print('hi')

    def __len__(self):
        return len(self._flat_breed_images)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target character class.
        """
        image_name, breed_class = self._flat_breed_images[index]
        image_path = join(self.images_folder, self._breeds[breed_class], image_name)
        image = Image.open(image_path).convert('RGB')

        if self.cropped:
            image = image.crop(self._flat_breed_annotations[index][1])

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            breed_class = self.target_transform(breed_class)

        return image, breed_class

    def download(self):
        import tarfile

        if os.path.exists(join(self.root, 'Images')) and os.path.exists(join(self.root, 'Annotation')):
            if len(os.listdir(join(self.root, 'Images'))) == len(os.listdir(join(self.root, 'Annotation'))) == 120:
                print('Files already downloaded and verified')
                return

        for filename in ['images', 'annotation']:
            tar_filename = filename + '.tar'
            url = self.download_url_prefix + '/' + tar_filename
            download_url(url, self.root, tar_filename, None)
            print('Extracting downloaded file: ' + join(self.root, tar_filename))
            with tarfile.open(join(self.root, tar_filename), 'r') as tar_file:
                tar_file.extractall(self.root)
            os.remove(join(self.root, tar_filename))

    def get_boxes(self, path):
        import xml.etree.ElementTree
        e = xml.etree.ElementTree.parse(path).getroot()
        boxes = []
        for objs in e.iter('object'):
            boxes.append([int(objs.find('bndbox').find('xmin').text),
                          int(objs.find('bndbox').find('ymin').text),
                          int(objs.find('bndbox').find('xmax').text),
                          int(objs.find('bndbox').find('ymax').text)])
        return boxes