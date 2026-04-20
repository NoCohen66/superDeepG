import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import sys
sys.path.append('../')
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

means = 0.1307
stddevs = 0.3081
img_shape = (1, 28, 28)
num_classes = 10

def get_dataset(path='./MNIST_Dataset', max_images=None):
    
    test_transform = transforms.ToTensor()
    test =  torchvision.datasets.MNIST(root=path, train=False, download=True, transform=test_transform)
    dataloader = DataLoader(test, batch_size=max_images, shuffle=False)
    data = next(iter(dataloader))
    images, labels = data
    images_final = []
    for image in images:
        image = image[None]
        images_final.append(image)
    return images_final, labels, 10, means, stddevs, img_shape


def get_single_test_image(image_index=0):
    test_transform = transforms.ToTensor()
    test_dataset = torchvision.datasets.MNIST(root='./MNIST_Dataset', train=False, download=True, transform=test_transform)
    single_image_dataset = Subset(test_dataset, [image_index])
    data_loader = DataLoader(single_image_dataset, batch_size=1, shuffle=False)
    single_image, label = next(iter(data_loader)) 
    return single_image, label