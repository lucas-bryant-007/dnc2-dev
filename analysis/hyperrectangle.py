# This file takes a frozen pretrained encoder and CelebA and selects three attributes
# using the CelebA training data. Then constructs their predicted 3D hyperrectangle,
# evaluate its eight centroids on held-out CelebA test data, and produce the figure and 
# also every number needed for progress report.

# for now should support the following models
# vicreg_celeba (local CelebA-trained checkpoint)
# ijepa_celeba (local CelebA-trained checkpoint)
# vicreg_imagenet (official Meta ImageNet weights)
# ijepa_imagenet (official Meta ImageNet weights)




# Steps:
# 1. Load encoder model
# 2. Extract CelebA features
# 3. Select 3 attributes using train only
# 4. Predict the eight +-sqrt(B_t) corners
# 5. Measure the eight held-out test centroids
# 6. Save one figure and one json report


def load_encoder(checkpoint):
    """
    Loads frozen encoder. Currently handles: 
    vicreg_celeba, ijepa_celeba, vicreg_imagenet, ijepa_imagenet
    """
    pass


import torch

print(torch.cuda.is_available())
print(torch.cuda.get_device_properties(0).name)

