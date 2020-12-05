from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

import os
import pickle
import numpy as np
from PIL import Image


# reference
# https://github.com/pytorch/vision/blob/master/torchvision/datasets/cifar.py
class SimSiamDataset(Dataset):
    base_folder = 'cifar-10-batches-py'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]
    meta = {
        'filename': 'batches.meta',
        'key': 'label_names',
        'md5': '5ff9c542aee3614f3951f8cda6e48888',
    }

    def __init__(self, args, mode='train', downstream=False):
        if mode == 'train':
            data_list = self.train_list
        else:
            data_list = self.test_list
        self.targets = []
        self.data = []
        self.args = args
        self.downstream = downstream

        for file_name, checksum in data_list:
            file_path = os.path.join(args.base_dir, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(self.args.img_size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(0.2),
            # transforms.GaussianBlur(kernel_size=int(self.args.img_size * 0.1), sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        if downstream:
            if mode == 'train':
                self.transform1 = train_transform
            else:
                self.transform1 = test_transform
        else:
            self.transform1 = train_transform
            self.transform2 = train_transform

    def __getitem__(self, index: int):
        img1, target = self.data[index], self.targets[index]
        img2 = img1.copy()

        img1 = Image.fromarray(img1)
        img1 = self.transform1(img1)

        if self.downstream:
            return img1, target

        img2 = Image.fromarray(img2)
        img2 = self.transform2(img2)

        return img1, img2, target

    def __len__(self) -> int:
        return len(self.data)


def load_data(args):
    train_data = SimSiamDataset(args)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    test_data = SimSiamDataset(args, mode='test')
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    down_train_data = SimSiamDataset(args, downstream=True)
    down_train_loader = DataLoader(down_train_data, batch_size=args.down_batch_size, shuffle=True, num_workers=args.num_workers)

    down_test_data = SimSiamDataset(args, mode='test', downstream=True)
    down_test_loader = DataLoader(down_test_data, batch_size=args.down_batch_size, shuffle=False, num_workers=args.num_workers)

    return train_loader, test_loader, down_train_loader, down_test_loader
