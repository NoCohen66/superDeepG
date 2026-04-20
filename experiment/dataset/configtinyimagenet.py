
import torchvision
import torchvision.transforms as transforms

import sys
sys.path.append('../')
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

means = (0.4819, 0.4457, 0.3934)
stddevs = (0.2734, 0.2650, 0.2770)
img_shape = (3, 56, 56)
num_classes = 200

def get_dataset(path='./tiny-imagenet-200', max_images=None):
    tfm = transforms.Compose([
        transforms.CenterCrop(img_shape[1]),  # ensure 56×56
        transforms.ToTensor()
    ])
    dataset = torchvision.datasets.ImageFolder(root=f"{path}/val", transform=tfm)
    
    images = []
    labels = []
    for i, (img, lbl) in enumerate(dataset):
        if i >= max_images:
            break
        images.append(img)
        labels.append(lbl)
    actual = len(images)
    if actual < max_images:
        print(f"Warning: only found {actual} images in {path}")
    
    return images, labels, num_classes, means, stddevs, img_shape