# code for transformations applied to custom dataset
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from skimage import filters


def CNNPreprocess():
    """
    Preprocess images for CNN model
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

def CNNPreprocessGrey():
    """
    Preprocess images for CNN model
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

def ResNetPreprocess():
    """
    Preprocess images for ResNet model
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

def ResNetPreprocessGrey():
    """
    Preprocess images for ResNet model
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

def ViTPreprocess():
    """
    Preprocess images for ViT model
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

def ViTPreprocessGrey():
    """
    Preprocess images for ViT model
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

class HessianFilterTransform(object):
    def __init__(self, sigmas=range(1, 10, 2)):
        self.sigmas = sigmas

    def __call__(self, img):
        img = np.array(img)
        img = filters.hessian(img, self.sigmas)
        img = Image.fromarray(img)
        return img

def ResNetPreprocessHessian():
    """
    Preprocess images for ResNet model
    *** ONLY USE IF YOU WANT TO APPLY HESSIAN FILTER ON THE FLY, THIS IS VERY SLOW ***
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=1),
        HessianFilterTransform(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])