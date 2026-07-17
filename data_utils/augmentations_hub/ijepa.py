from torchvision import transforms
from data_utils.augmentations_hub.common_transforms import RepeatChannelsIfNeeded


def get_ijepa_transforms(dataset: str = "imagenet", img_size: int = 224):
    """
    Returns data augmentation (train) and basic evaluation transforms for a given dataset.

    Args:
        dataset (str): Name of the dataset. Options: 'imagenet', 'cifar', 'svhn'.

    Returns:
        train_transform (torchvision.transforms.Compose): Data augmentation pipeline.
        basic_transform (torchvision.transforms.Compose): Standard test/eval pipeline.
    """
    dataset = dataset.lower()
    
    if 'imagenet' in dataset:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            RepeatChannelsIfNeeded(),
            transforms.Normalize(mean=mean, std=std),
        ])
        basic_transform = transforms.Compose([
            transforms.Resize(round(img_size * 256 / 224)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            RepeatChannelsIfNeeded(),
            transforms.Normalize(mean=mean, std=std),
        ])

    elif dataset == "celeba":
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        train_transform = transforms.Compose([
            transforms.RandomCrop(160),
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            RepeatChannelsIfNeeded(),
            transforms.Normalize(mean=mean, std=std),
        ])
        basic_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            RepeatChannelsIfNeeded(),
            transforms.Normalize(mean=mean, std=std),
        ])

    elif 'cifar' in dataset or dataset == 'svhn':
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        basic_transform = transforms.Compose([
                transforms.Resize(img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ])
    else:
        raise NotImplementedError(f"Unknown dataset: {dataset}")

    return train_transform, basic_transform
