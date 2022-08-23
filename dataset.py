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
    training_size = 60000, 
    validation_size = 5000,
    test_size = 10000,
    root = "/home/krishna/Documents/mas_sem_5/dlrv"):
    
    """ 
    Reads and converts data from the MNIST/cifar10 dataset. 
     Args:
        dataset : 'mnist' or 'cifar10'
        training_size (int) :  number of images from training size
        validation_size (int) : number of images from validation size
        test_size (int) : number of images from test_size
    Returns:
         x_train, y_train, x_test, y_test, class_names    
         x_train, x_test: training and test images
        y_train, y_test: corresponding labels 
        class_names: array with names of classes
    """
    np.random.seed(4711)
    
    if dataset == 'mnist':
        trainset = torchvision.datasets.MNIST(root=f"{root}/mnist_data", train=True, download=True,
                                                transform=torchvision.transforms.ToTensor(),)                                        
        train_subset, val_subset = torch.utils.data.random_split(trainset, [training_size-validation_size, validation_size])
        trainloader = torch.utils.data.DataLoader(train_subset, batch_size=8, shuffle=True, num_workers=2)
        x_train, y_train = extract_data_from_loader(trainloader)
        x_train = x_train.transpose([0, 2, 3, 1]).squeeze()
        x_train = (x_train * 255).astype(np.uint8)

        valloader = torch.utils.data.DataLoader(val_subset, batch_size=8, shuffle=True, num_workers=2)
        x_val, y_val = extract_data_from_loader(valloader)
        x_val = x_val.transpose([0, 2, 3, 1]).squeeze()
        x_val = (x_val * 255).astype(np.uint8)

        testset = torchvision.datasets.MNIST(root=f"{root}/mnist_data", train=False, download=True,
                                                transform=torchvision.transforms.ToTensor())                                          
        testloader = torch.utils.data.DataLoader(testset, batch_size=8, shuffle=True, num_workers=2)
        x_test, y_test = extract_data_from_loader(testloader)
        x_test = x_test.transpose([0, 2, 3, 1]).squeeze()
        x_test = (x_test * 255).astype(np.uint8)
        class_ids = np.unique(y_train)
        class_names = list(map(str, class_ids))

    elif dataset == 'cifar10':
        
        trainset = torchvision.datasets.CIFAR10(root=f"{root}/cifar10_data", train=True, download=True,
                                            transform = torchvision.transforms.ToTensor())
        train_subset, val_subset = torch.utils.data.random_split(trainset, [training_size-validation_size, validation_size])
        trainloader = torch.utils.data.DataLoader(train_subset, batch_size=8, shuffle=True, num_workers=2)
        x_train, y_train = extract_data_from_loader(trainloader)
        x_train = x_train.transpose([0, 2, 3, 1]).squeeze()
        x_train = (x_train * 255).astype(np.uint8)

        valloader = torch.utils.data.DataLoader(val_subset, batch_size=8, shuffle=True, num_workers=2)
        x_val, y_val = extract_data_from_loader(valloader)
        x_val = x_val.transpose([0, 2, 3, 1]).squeeze()
        x_val = (x_val * 255).astype(np.uint8)

        testset = torchvision.datasets.CIFAR10(root=f"{root}/cifar10_data", train=False, download=True,
                                            transform = torchvision.transforms.ToTensor())
        testloader = torch.utils.data.DataLoader(testset, batch_size=8, shuffle=True, num_workers=2)
        x_test, y_test = extract_data_from_loader(testloader)
        x_test = x_test.transpose([0, 2, 3, 1]).squeeze()
        x_test = (x_test * 255).astype(np.uint8)

        class_ids = np.unique(y_test)
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        class_names = [class_names[idx] for idx in class_ids]
            
        
    else:
        raise ValueError('The variable dataset must either be set to "mnist" or "cifar10"')

    x_train = x_train.astype( np.float32 )
    x_test = x_test.astype( np.float32 )

    return x_train, y_train, x_test, y_test,x_val,y_val, class_names

if __name__ == '__main__':

    x_train, y_train, x_test, y_test,x_val,y_val, class_names = get_dataset(dataset='mnist', training_size=6000,validation_size=1000, test_size=1000)
    print(class_names)
    # Show one example of each class 
    plt.figure()
    for class_id in range(len(class_names)):
        plt.subplot(2, 5, class_id + 1)
        plt.imshow(x_train[y_train == class_id][0].astype(np.uint8))
        plt.title(class_names[class_id])
    plt.show()    