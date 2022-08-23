

import numpy as np
try:
    import torch
    import torchvision
    using_tf = False
except:
    from tensorflow.keras.datasets import mnist, cifar10
    using_tf = True
import matplotlib.pyplot as plt

def cull_unused_classes(x, y, used_labels):
    """
    Removes data x and label y entries for classes not in used_labels. 
    """    
    idxs = [ label in used_labels for label in y ]
    x = x[idxs]
    y = y[idxs]

    return x, y

def make_successive_labels(y):
    """
    Replaces the integers in y such that only successive integers appear. 

    Example: [2 4 4 2 6 2 9 9 4 2] -> [0 1 1 0 2 0 3 3 1 0]
    """
    for (new_label, unique_label) in enumerate(np.unique(y)):
        y[y == unique_label] = new_label    

    return y

def extract_data_from_loader(loader):
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
        if using_tf:
            (x_train, y_train),(x_test, y_test) = mnist.load_data() # x_train.shape = 50,000 x 28 x 28
        else:
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
        if using_tf:
            (x_train, y_train),(x_test, y_test) = cifar10.load_data() # x_train.shape = 50,000 x 32 x 32 x 3
            y_train = y_train.reshape([-1])
            y_test = y_test.reshape([-1])
        else:
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

    if num_classes != 10:
        x_train, y_train = cull_unused_classes(x_train, y_train, used_labels)
        x_test, y_test = cull_unused_classes(x_test, y_test, used_labels)

    # make the class indices consecutive (only if not all 10 digits are chosen in used_labels)

    y_train = make_successive_labels(y_train)
    y_test = make_successive_labels(y_test)

    original_training_size = x_train.shape[0]
    original_test_size = x_test.shape[0]

    # take a small random subset of images (size is given in training_size and test_size)
    training_idxs = np.arange(original_training_size)
    np.random.shuffle(training_idxs)
    training_idxs = training_idxs[0:training_size]
    x_train = x_train[training_idxs]
    y_train = y_train[training_idxs]
    y_train = y_train.astype(np.int32)

    test_idxs = np.arange(original_test_size)
    np.random.shuffle(test_idxs)
    test_idxs = test_idxs[0:test_size]
    x_test = x_test[test_idxs]
    y_test = y_test[test_idxs]
    y_test = y_test.astype(np.int32)

    x_train = x_train.astype( np.float32 )
    x_test = x_test.astype( np.float32 )

    return x_train, y_train, x_test, y_test, class_names

if __name__ == '__main__':

    x_train, y_train, x_test, y_test, class_names = get_dataset(dataset='cifar10', training_size=600, test_size=100)

    print(x_train.shape)

    # Show one example of each class 
    plt.figure()
    for class_id in range(len(class_names)):
        plt.subplot(2, 5, class_id + 1)
        plt.imshow(x_train[y_train == class_id][0].astype(np.uint8)) # plotting behaviour of imshow differs between float32 and uint8
        plt.title(class_names[class_id])
    plt.show()    