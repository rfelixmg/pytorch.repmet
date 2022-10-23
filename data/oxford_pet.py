from __future__ import print_function

from PIL import Image
from os.path import join
import os
import scipy.io

import configs

import torch.utils.data as data
from torchvision.datasets.utils import download_url, list_dir, list_files


class OxPet(data.Dataset):
    """`Oxford-IIIT Pet <http://www.robots.ox.ac.uk/~vgg/data/pets/>`_ Dataset.
    Args:
        root (string): Root directory of dataset where directory exists.
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
    folder = 'OxfordPets'
    download_url_prefix = 'http://www.robots.ox.ac.uk/~vgg/data/pets/data'

    def __init__(self,
                 root,
                 train=True,
                 cropped=False,
                 transform=None,
                 target_transform=None,
                 download=False):

        self.root = join(os.path.expanduser(root), self.folder)
        self.train = train
        self.cropped = cropped
        self.transform = transform
        self.target_transform = target_transform

        if download:
            self.download()

        self.images_folder = join(self.root, 'images')
        self.annotations_folder = join(self.root, 'annotations')

        split = self.load_split()

        if self.cropped:
            self.annotations = [[(annotation, box, idx)
                                for box in self.get_boxes(join(self.annotations_folder, annotation))]
                                for annotation, idx in split]
            self.flat_annotations = sum(self.annotations, [])

            self.flat_images = [(annotation+'.jpg', idx) for annotation, box, idx in self.flat_annotations]
        else:
            self.images = [(annotation+'.jpg', idx) for annotation, idx in split]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target character class.
        """
        image_name, target_class = self.images[index]
        image_path = join(self.images_folder, image_name)
        image = Image.open(image_path).convert('RGB')

        if self.cropped:
            image = image.crop(self.annotations[index][1])

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            target_class = self.target_transform(target_class)

        return image, target_class

    def download(self):
        import tarfile

        if os.path.exists(join(self.root, 'images')) and os.path.exists(join(self.root, 'annotations')):
            if len(os.listdir(join(self.root, 'images'))) == 7393 and len(os.listdir(join(self.root, 'annotations', 'xmls'))) == 3686:
                # images should really be 7390 as 3 are .mat files ... why? :/
                print('Files already downloaded and verified')
                return

        for filename in ['images', 'annotations']:
            tar_filename = filename + '.tar.gz'
            url = self.download_url_prefix + '/' + tar_filename
            download_url(url, self.root, tar_filename, None)
            print('Extracting downloaded file: ' + join(self.root, tar_filename))
            with tarfile.open(join(self.root, tar_filename), 'r') as tar_file:
                def is_within_directory(directory, target):
                    
                    abs_directory = os.path.abspath(directory)
                    abs_target = os.path.abspath(target)
                
                    prefix = os.path.commonprefix([abs_directory, abs_target])
                    
                    return prefix == abs_directory
                
                def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
                
                    for member in tar.getmembers():
                        member_path = os.path.join(path, member.name)
                        if not is_within_directory(path, member_path):
                            raise Exception("Attempted Path Traversal in Tar File")
                
                    tar.extractall(path, members, numeric_owner=numeric_owner) 
                    
                
                safe_extract(tar_file, self.root)
            os.remove(join(self.root, tar_filename))

    @staticmethod
    def get_boxes(path):
        boxes = []
        if os.path.exists(path):
            import xml.etree.ElementTree
            e = xml.etree.ElementTree.parse(path).getroot()

            for objs in e.iter('object'):
                boxes.append([int(objs.find('bndbox').find('xmin').text),
                              int(objs.find('bndbox').find('ymin').text),
                              int(objs.find('bndbox').find('xmax').text),
                              int(objs.find('bndbox').find('ymax').text)])
        return boxes

    def load_split(self):
        if self.train:
            with open(join(self.annotations_folder, 'trainval.txt')) as f:  # 3680 samples
                lines = f.readlines()
        else:
            with open(join(self.annotations_folder, 'test.txt')) as f:  # 3699 samples
                lines = f.readlines()

        lines = [line.split()[0:2] for line in lines]
        split = [line[0] for line in lines]
        labels = [int(line[1]) for line in lines]

        return list(zip(split, labels))

    def stats(self):
        counts = {}
        for index in range(len(self.images)):
            image_name, target_class = self.images[index]
            if target_class not in counts.keys():
                counts[target_class] = 1
            else:
                counts[target_class] += 1

        print("%d samples spanning %d classes (avg %f per class)"%(len(self.images), len(counts.keys()), float(len(self.images))/float(len(counts.keys()))))

        return counts


if __name__ == "__main__":
    c = OxPet(os.path.join(configs.general.paths.imagesets), download=True)