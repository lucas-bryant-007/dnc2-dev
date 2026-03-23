from data_utils.augmentations_hub import ijepa, vicreg

AUGMENTATION_REGISTRY = {
    'ijepa': ijepa.get_ijepa_transforms,
    'vicreg': vicreg.get_vicreg_transforms,

}

def get_transforms(method, dataset='imagenet'):
    key = method.lower()
    if key not in AUGMENTATION_REGISTRY:
        raise NotImplementedError(f"No augmentations defined for {method}")
    return AUGMENTATION_REGISTRY[key](dataset)
