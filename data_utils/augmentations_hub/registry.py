from data_utils.augmentations_hub import ijepa, vicreg, wmse

AUGMENTATION_REGISTRY = {
    'ijepa': ijepa.get_ijepa_transforms,
    'vicreg': vicreg.get_vicreg_transforms,
    'wmse': wmse.get_wmse_transforms,
}

def get_transforms(method, dataset="imagenet", img_size=224):
    key = method.lower()
    if key not in AUGMENTATION_REGISTRY:
        raise NotImplementedError(f"No augmentations defined for {method}")
    return AUGMENTATION_REGISTRY[key](dataset, img_size=img_size)
