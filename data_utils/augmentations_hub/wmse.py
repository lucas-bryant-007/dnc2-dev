from torchvision import transforms
from data_utils.augmentations_hub.common_transforms import RepeatChannelsIfNeeded

def get_wmse_transforms(dataset: str = "imagenet", img_size: int = 224):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.08, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),

        transforms.RandomApply([
            transforms.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.2,
                hue=0.1,
            )
        ], p=0.8),

        transforms.RandomGrayscale(p=0.2),

        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=23)
        ], p=0.5),

        transforms.ToTensor(),
        RepeatChannelsIfNeeded(),
        transforms.Normalize(mean=mean, std=std),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize(round(img_size * 256 / 224)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        RepeatChannelsIfNeeded(),
        transforms.Normalize(mean=mean, std=std),
    ])

    return train_transform, eval_transform
