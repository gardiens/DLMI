
import torchvision.transforms as T
import torchvision.transforms.v2 as T2  # for more complex transforms (PyTorch ≥ 0.15)
from PIL import Image
import random
import torch

class Random90OrTranspose:
    def __call__(self, img):
        choice = random.random()
        if choice < 0.25:
            return img.transpose(Image.ROTATE_90)
        elif choice < 0.5:
            return img.transpose(Image.ROTATE_270)
        elif choice < 0.75:
            return img.transpose(Image.TRANSPOSE)
        else:
            return img
        return img

tta_transform = T.Compose([
    T.ToPILImage(),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomApply([
        T.RandomAutocontrast(),  # approximation for CLAHE
        T.RandomAdjustSharpness(sharpness_factor=2),
        T.RandomAdjustSharpness(sharpness_factor=0.5),  # as an IAAEmboss approx
        T.ColorJitter(brightness=0.2),
        T.ColorJitter(contrast=0.2),
        T.GaussianBlur(kernel_size=3),
    ], p=0.5),
    T.ColorJitter(hue=0.2, saturation=0.2),
    T.RandomAffine(degrees=45, translate=(0.15, 0.15), scale=(0.85, 1.15)),
    T.ToTensor(),
    T.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])  # match albumentations.Normalize()
])
def get_transform_tta(tta_strategy):
    if tta_strategy=="densenet169":
        tta_transform = T.Compose([
            # convert to PIL Image
    T.ToPILImage(),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomApply([
        T.RandomAutocontrast(),  # approximation for CLAHE
        T.RandomAdjustSharpness(sharpness_factor=2),
        T.RandomAdjustSharpness(sharpness_factor=0.5),  # as an IAAEmboss approx
        T.ColorJitter(brightness=0.2),
        T.ColorJitter(contrast=0.2),
        T.GaussianBlur(kernel_size=3),
    ], p=0.5),
    T.ColorJitter(hue=0.2, saturation=0.2),
    T.RandomAffine(degrees=45, translate=(0.15, 0.15), scale=(0.85, 1.15)),
    T.ToTensor(),
    T.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])  # match albumentations.Normalize()
    , 
    ])
    elif tta_strategy=="NoTTA":
        tta_transform = T.Lambda(lambda x: x)
    else:
        raise NotImplementedError(f"TTA strategy {tta_strategy} not implemented")
    return tta_transform
def apply_tta(sample,num_tta:int,tta_transform):
    #? We suppose TTA to be a tensor here 
    # create a output tensor of size (num_tta,sample.shape)

    output=torch.zeros((num_tta,sample.shape[0],sample.shape[1],sample.shape[2]))
    
    # the zero one is the original one
    output[0]=sample
    for i in range(1,num_tta):
        output[i]=tta_transform(sample)
    return output