
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def extract_data_from_loader(loader):
    """
    Function extracts data from loader

    Args:
        loader : pytorch data oader

    Returns:
        x,y : image and label of data
    """
    x = np.array([])
    y = np.array([])
    for curr_batch in loader:
        if x.shape[0] == 0:
            x = curr_batch[0]
            y = curr_batch[1]
        else:
            x = np.concatenate([x, curr_batch[0]], axis=0)
            y = np.concatenate([y, curr_batch[1]], axis=0)
    return x, y

def get_dataset( dataset = 'mnist', 
    used_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 
    training_size = 60000, 
    test_size = 10000,
    root ="./mnist/"):
    """ 
    Reads and converts data from the MNIST dataset. 

    :param  dataset         'mnist' or 'cifar10'
    :param  used_labels     list of digit classes to include into the returned subset
    :param  training_size   number of images from training size
    :param  test_size       number of images from test_size

    :return x_train, y_train, x_test, y_test, class_names    
      x_train, x_test: training and test images (uint8 [0- 255], shape: training_size x 28 x 28, test_size x 28 x 28),
      y_train, y_test: corresponding labels (int32 [0 - len(used_labels)), shape: training_size / test_size)
      class_names: array with names of classes (size: len(used_labels))
    """

    num_classes = len(used_labels)
    max_num_classes = 10


    np.random.seed(4711)

    if dataset == 'mnist':
        trainset = torchvision.datasets.MNIST(root=f"{root}/mnist_data", train=True, download=True,
                                              transform=torchvision.transforms.ToTensor())
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=training_size, shuffle=False, num_workers=2)
        x_train, y_train = extract_data_from_loader(trainloader)
        x_train = x_train.transpose([0, 2, 3, 1]).squeeze()
        x_train = (x_train * 255).astype(np.uint8)

        testset = torchvision.datasets.MNIST(root=f"{root}/mnist_data", train=False, download=True,
                                              transform=torchvision.transforms.ToTensor())
        testloader = torch.utils.data.DataLoader(testset, batch_size=test_size, shuffle=False, num_workers=2)
        x_test, y_test = extract_data_from_loader(testloader)
        x_test = x_test.transpose([0, 2, 3, 1]).squeeze()
        x_test = (x_test * 255).astype(np.uint8)

        class_names = list(map(str, used_labels))

    elif dataset == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(root=f"{root}/cifar10_data", train=True, download=True,
                                            transform = torchvision.transforms.ToTensor())
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=training_size, shuffle=False, num_workers=2)
        x_train, y_train = extract_data_from_loader(trainloader)
        x_train = x_train.transpose([0, 2, 3, 1]).squeeze()
        x_train = (x_train * 255).astype(np.uint8)

        testset = torchvision.datasets.CIFAR10(root=f"{root}/cifar10_data", train=False, download=True,
                                            transform = torchvision.transforms.ToTensor())
        testloader = torch.utils.data.DataLoader(testset, batch_size=test_size, shuffle=False, num_workers=2)
        x_test, y_test = extract_data_from_loader(testloader)
        x_test = x_test.transpose([0, 2, 3, 1]).squeeze()
        x_test = (x_test * 255).astype(np.uint8)

        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        class_names = [class_names[idx] for idx in used_labels]
            
        
    else:
        raise ValueError('The variable dataset must either be set to "mnist" or "cifar10"')

    x_train = x_train.astype( np.float32 )
    x_test = x_test.astype( np.float32 )

    return x_train, y_train, x_test, y_test, class_names

if __name__ == '__main__':

    x_train, y_train, x_test, y_test,x_val,y_val, class_names = get_dataset(dataset='mnist',training_size=6000,validation_size=1000, test_size=1000)
    print(class_names)
    # Show one example of each class 
    plt.figure()
    for class_id in range(len(class_names)):
        plt.subplot(2, 5, class_id + 1)
        plt.imshow(x_train[y_train == class_id][0].astype(np.uint8))
        plt.title(class_names[class_id])
    plt.show()    