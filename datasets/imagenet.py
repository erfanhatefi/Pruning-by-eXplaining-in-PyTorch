import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from datasets.imagenet_classnames import CLASS_NAMES


class ImageNetSubset(Dataset):
    """
    Generate a subset of the dataset given the indices of the samples
    """

    def __init__(self, dataset, indices):
        """
        Constructor of the class

        Args:
            dataset (torch.utils.data.Dataset): a dataset
            indices (list): indices of the samples from the original dataset
        """
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]


class ImageNet:

    def __init__(self, path, random_seed, num_workers=8):
        """
        Constructor of the class
        Can get the dataset from https://www.image-net.org/download.php

        Args:
            path (str): path to the dataset
            random_seed (int): random seed
            num_workers (int, optional): number of workers. Defaults to 8.
        """
        self.path = path
        self.num_workers = num_workers

        # Set the random seed
        torch.random.manual_seed(random_seed)

    def get_train_set(self):
        """
        Get the train dataset

        Returns:
            torch.utils.data.dataset.Subset: train set
        """
        train_set = ImageNetDataset(
            self.path, train=True, transform=self.train_transformation()
        )

        return train_set

    def get_valid_set(self):
        """
        Get the validation dataset

        Returns:
            torch.utils.data.dataset.Subset: validation set
        """
        validation_set = ImageNetDataset(
            self.path, train=False, transform=self.validation_transformation()
        )

        return validation_set

    def train_transformation(self):
        """
        Transformation for training data

        Returns:
            torchvision.transforms: transformation for training data
        """
        return transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def validation_transformation(self):
        """
        Transformation for validation data

        Returns:
            torchvision.transforms: transformation for validation data
        """
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((224, 224)),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )


class ImageNetDataset(datasets.ImageFolder):
    def __init__(
        self,
        root,
        *args,
        validate=False,
        train=True,
        use_precomputed_labels=False,
        labels_path=None,
        transform=None,
        **kwargs,
    ):
        """
        ImageNet root folder is expected to have two directories: train and val.

        Args:
            root (str): root directory
            validate (bool, optional): is validate set or not. Defaults to False.
            train (bool, optional): is train set or not. Defaults to True.
            use_precomputed_labels (bool, optional): _description_. Defaults to False.
            labels_path (_type_, optional): _description_. Defaults to None.
            transform (_type_, optional): _description_. Defaults to None.
        """

        if train and validate == train:
            raise ValueError("Train and validate can not be True at the same time.")
        if use_precomputed_labels and labels_path is None:
            raise ValueError(
                "If use_precomputed_labels=True the labels_path is necessary."
            )

        if train:
            root = os.path.join(root, "train")
        elif validate:
            root = os.path.join(root, "val_mpeg")
        else:
            root = os.path.join(root, "val")

        super().__init__(root, transform=transform, *args, **kwargs)
        self.transforms = transform
        self.preprocessing = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        if validate and use_precomputed_labels:
            df = pd.read_csv(labels_path, sep="\t")
            df.input_path = df.input_path.apply(lambda x: os.path.join(root, x))
            mapping = dict(zip(df.input_path, df.pred_class))
            self.samples = [(x[0], mapping[x[0]]) for x in self.samples]
            self.targets = [x[1] for x in self.samples]

        self.class_names = list(CLASS_NAMES.values())
        self.num_classes = len(self.class_names)

    def reverse_augmentation(self, data: torch.Tensor):
        """
        Reverse the augmentation from Normalization as
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        Args:
            data (torch.Tensor): a given tensor

        Returns:
            torch.tensor: unnormalized tensor
        """
        data = data.clone() + 0
        mean = torch.Tensor(self.preprocessing.mean).to(data)
        var = torch.Tensor(self.preprocessing.std).to(data)
        data *= var[:, None, None]
        data += mean[:, None, None]
        return torch.multiply(data, 255)

    @staticmethod
    def class_id_name(class_id: int):
        """
        Get the class name based on the class id

        Args:
            class_id (int): class_id

        Returns:
            str: name of the class
        """
        return CLASS_NAMES[class_id]

    def get_target(self, index):
        """
        Get the target based on the index

        Args:
            index (int): index
            torch.tensor: target
        """
        return self.targets[index]

    def random_classes(self, num_classes=3):
        """
        Generate random classes via torch.randperm which generates a
        random permutation of integers from 0 to num_classes

        Args:
            num_classes (int, optional): number of random classes. Defaults to 3.

        Returns:
            torch.tensor: the generated permutation(as random classes)
        """
        return torch.randperm(self.num_classes)[:num_classes]


def get_sample_indices_for_class(
    dataset, list_classes, num_samples_per_class=5, device="cpu"
):
    """
    Get indices of samples based on the chosen classes

    Args:
        dataset (torch.utils.data.Dataset): dataset
        list_classes (list): list of classes to choose from
        num_samples_per_class (int, optional): The number of samples
                            to choose from each class. Defaults to 5.
        device (str, optional): device. Defaults to "cpu".

    Returns:
        list: list of indices from the given dataset based on the chosen classes
    """
    indices = torch.Tensor([]).to(device)
    for i, c in enumerate(list_classes):
        # Get all of the indices for the specific class
        class_indices = torch.where(
            torch.tensor(dataset.targets).to(device) == torch.tensor(c).to(device)
        )[0].to(device)
        if num_samples_per_class != "all":
            if num_samples_per_class <= dataset.targets.count(c):
                # Choose a random subset of the indices
                random_indices_for_class = torch.randperm(len(class_indices))[
                    :num_samples_per_class
                ].to(device)
                # Get the indices for the class chosen in a random order
                class_indices = class_indices[random_indices_for_class].to(device)
                # Append the indices to the list
        indices = torch.cat((indices, class_indices), dim=0)

    indices = indices.long()
    # change indices to list of integers
    indices = indices.cpu().tolist()
    return indices


def get_imagenet(data_path, split="train", preprocessing=True):
    """
    Get the ImageNet dataset given the path and the split desired

    Args:
        data_path (str): path
        split (str, optional): The split desired. Defaults to "train".
        preprocessing (bool, optional): whether to apply preprocessing or not. Defaults to True.

    Returns:
        _type_: _description_
    """
    if split == "train":
        transform = all_transforms["imagenet"]["train"]
    else:
        transform = all_transforms["imagenet"]["val"]
    if not preprocessing:
        transform = transforms.Compose(
            [t for t in transform.transforms if not isinstance(t, transforms.Normalize)]
        )
    return ImageNet(data_path, train=split == "train", transform=transform)


all_transforms = {
    "imagenet": {
        "train": transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        ),
    }
}
