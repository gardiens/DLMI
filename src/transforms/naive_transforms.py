import torchvision.transforms as transforms

transform = transforms.Compose(
    [
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

transform_train = transforms.Compose(
    [
        transforms.Resize(
            (32, 32)
        ),  # resises the image so it can be perfect for our model.
        transforms.RandomHorizontalFlip(),  # FLips the image w.r.t horizontal axis
        transforms.RandomRotation(10),  # Rotates the image to a specified angel
        transforms.RandomAffine(
            0, shear=10, scale=(0.8, 1.2)
        ),  # Performs actions like zooms, change shear angles.
        transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2
        ),  # Set the color params
        transforms.ToTensor(),  # comvert the image to tensor so that it can work with torch
        transforms.Normalize(
            (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
        ),  # Normalize all the images
    ]
)
import albumentations as A
import torch
import numpy as np
from torchvision import transforms
import torchvision.transforms.v2 as T 
def get_transform(transform_name:str="basic",resolution=98,stage="train"):
    transform=None
    if transform_name=="basic":
        transform=transforms.Resize((resolution,resolution))
    elif transform_name=="opti_A":
        if stage=="train":
            p=1
            transform=A.Compose([
            A.Resize(resolution, resolution),
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.RandomRotate90(),
            A.Transpose(),
            A.Affine(
                scale=(0.5, 1.5),
                translate_percent=(0.0, 0.0625),
                rotate=(-45, 45),
                p=0.75
            ),
            A.OpticalDistortion(),
            A.GridDistortion(),
            A.RandomBrightnessContrast(p=0.3),
            A.RandomGamma(p=0.3),
            A.OneOf([
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=0.1, val_shift_limit=0.1, p=0.3),
                A.ChannelShuffle(p=0.3),
                A.CLAHE(p=0.3),
            ])
        ], p=p)
        else:
            transform=A.Compose([
            A.Resize(resolution, resolution),
        ], p=1)
    elif transform_name=="opti":
        transform= T.Compose([
        T.Resize((resolution, resolution)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomRotation((0, 90)),  # torchvision does not support strict 90-degree rotations or transpose
        T.RandomAffine(
            degrees=45,
            translate=(0.0, 0.0625),
            scale=(0.5, 1.5),
        ),
        T.ColorJitter(brightness=0.3, contrast=0.3),
        T.RandomApply([
            T.ColorJitter(hue=0.1, saturation=0.1),
        ], p=0.3),
        transforms.ToTensor()
    ])
    else:
        print("no passe par la?")
        raise ValueError(f"Transform {transform_name} not found")

    return transform