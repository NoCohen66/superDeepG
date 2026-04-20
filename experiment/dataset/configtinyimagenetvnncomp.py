
import torchvision
import torchvision.transforms as transforms
import torch
import sys
sys.path.append('../')
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import os
means = (0.4802, 0.4481, 0.3975)
stddevs = (0.2302, 0.2265, 0.2262)
img_shape = (3, 56, 56)
num_classes = 200


def get_dataset(path='./tiny-imagenet-200', max_images=None):
    #normalize = transforms.Normalize(mean=means, std=stddevs)
    
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Tiny ImageNet dataset not found at: '{data_path}'.\n"
            "Please download it and ensure the path is correct in the script.")

    val_dir = os.path.join(path, 'val')
    val_img_dir = os.path.join(val_dir, 'images')

    if os.path.exists(val_img_dir):
        print("INFO: Reorganizing Tiny ImageNet validation folder structure (one-time operation)...")
        annotations_file = os.path.join(val_dir, 'val_annotations.txt')
        with open(annotations_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                img_name, class_name = parts[0], parts[1]
                class_dir = os.path.join(val_dir, class_name)
                os.makedirs(class_dir, exist_ok=True)
                src_path = os.path.join(val_img_dir, img_name)
                dest_path = os.path.join(class_dir, img_name)
                if os.path.exists(src_path):
                    os.rename(src_path, dest_path)
        os.rmdir(val_img_dir) 
        os.remove(annotations_file)
        print("INFO: Structure fixed.")

    test_dataset = torchvision.datasets.ImageFolder(val_dir, transform=transforms.Compose([transforms.CenterCrop(56),transforms.ToTensor()]))#, normalize]))
    test_loader = torch.utils.data.DataLoader(test_dataset, pin_memory=True, num_workers=4)
    images = []
    labels = []
    for i, (img, lbl) in enumerate(test_loader):
        if i >= max_images:
            break
        images.append(img)
        labels.append(lbl)
    actual = len(images)
    if actual < max_images:
        print(f"Warning: only found {actual} images in {path}")
    
    return images, labels, num_classes, means, stddevs, img_shape