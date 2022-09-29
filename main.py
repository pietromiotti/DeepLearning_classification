import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
#from tensorboardX import SummaryWriter
import random as random

#uncomment if you would like to monitor in live performance
#writer = SummaryWriter(logdir='my_experiment/')



#fix seed for reproducibility
random.seed(0)

#DEFINE MACRO
CHANNEL_R = 0
CHANNEL_G = 1
CHANNEL_B = 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyper-parameters
input_size = 3
output_size = 10

# for training:
batch_size = 32
learning_rate = 0.001
momentum = 0.9
total_epoch_no_dropout = 20
total_epoch_dropout = 30


'''
Util function implemented to plot correctly an image from the CIFAR10 dataset
    input: instance of the dataset
    output: None, just plot the image

'''
def showImg(img, block=True, id=0):
    plt.figure(id)
    plt.imshow(np.transpose(img.numpy(), (1, 2, 0)))
    plt.show(block=block)


'''
Util function implemented to get the mean and the std of the dataset
    input: dataset
    output: mean and stds of the dataset

'''
def getMeanAndStd(set):
    set = [elem[0] for elem in set]
    set = torch.stack(set, dim=0).numpy()

    mean_r = set[:, CHANNEL_R, :, :].mean()
    mean_g = set[:, CHANNEL_G, :, :].mean()
    mean_b = set[:, CHANNEL_B, :, :].mean()
    means_set = [mean_r, mean_g, mean_b]

    std_r = set[:, CHANNEL_R, :, :].std()
    std_g = set[:, CHANNEL_G, :, :].std()
    std_b = set[:, CHANNEL_B, :, :].std()
    stds_set= [std_r, std_g, std_b]

    return means_set, stds_set


'''EXERCISE 1.1.1, IMPORT TRAINING AND TESTING SET '''
train_set = torchvision.datasets.CIFAR10(
    root='./data', train=True, transform=transforms.ToTensor(), download=True)

test_set = torchvision.datasets.CIFAR10(
    root='./data', train=False ,transform=transforms.ToTensor())


'''EXERCISE 1.1.2, PREPROCESSING TRAINING AND TESTING SET '''

'PreProcessing of the trainig Set - Normalization of the channels'
means_train, stds_train = getMeanAndStd(train_set)

normalize_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(means_train, stds_train)
])

train_set_normalized = torchvision.datasets.CIFAR10(
    root='./data', train=True, transform=normalize_train, download=False)

'PreProcessing of the test Set - Normalization of the channels'

test_set_normalized = torchvision.datasets.CIFAR10(
    root='./data', train=False, transform=normalize_train, download=False)

#Compute the mean for train set and test set
test_loader = torch.utils.data.DataLoader(
    dataset=test_set_normalized, batch_size=batch_size, shuffle=False)

'''EXERCISE 1.1.3, CREATING VALIDATION SET '''

idx = np.arange(len(train_set_normalized))

VALIDATION_SET_LENGHT = 1000

val_indices = idx[len(train_set_normalized)-VALIDATION_SET_LENGHT:]
train_indices= idx[:-VALIDATION_SET_LENGHT]

train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
valid_sampler = torch.utils.data.SubsetRandomSampler(val_indices)

train_loader = torch.utils.data.DataLoader(train_set_normalized, batch_size=batch_size,
                                          sampler=train_sampler, num_workers=2)

valid_loader = torch.utils.data.DataLoader(train_set_normalized, batch_size=batch_size,
                                          sampler=valid_sampler, num_workers=2)



'''EXERCISE 1.2 - MODEL IMPLEMENTATION'''

class convNet(nn.Module):
    def __init__(self, input_size, num_classes, apply_dropout=True):
        super(convNet, self).__init__()
        self.conv1 = nn.Conv2d(input_size, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3)
        self.pool1 = nn.MaxPool2d(2,2)
        self.conv3 = nn.Conv2d(32,64,kernel_size=3)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64*5*5,512)
        self.dropout = apply_dropout
        self.dropout_layer = nn.Dropout(0.2)
        self.fc2 = nn.Linear(512, num_classes)
        #The softmax layer is not present in the model since it is already implicitly defined in the CrossEntropyLoss function

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(F.relu(self.conv2(x)))
        if (self.dropout):
            x = self.dropout_layer(x)
        x = F.relu(self.conv3(x))
        x = self.pool2(F.relu(self.conv4(x)))
        if (self.dropout):
            x = self.dropout_layer(x)
        x = x.reshape(-1, 64*5*5)
        x = F.relu(self.fc1(x))
        
        if(self.dropout):
            x = self.dropout_layer(x)
        x = F.relu(self.fc2(x))

        return x

'''
Function used to train and validate the model
    input : model
    output:
            trainingAccuracy: vector containing training accuracy foreach epoch
            validationAccuracy: vector containing validation accuracy foreach epoch
            trainingLoss : vector containing training loss foreach epoch
            validationLoss: vector containing validation loss foreach epoch
'''


def train_and_validate(model, dropout=True):
    loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    maxValidAccuracy = 0
    epoch_maxValidAccuracy = 0

    trainingAccuracy = []
    validationAccuracy = []
    trainingLoss = []
    validationLoss = []

    # since dropout can slow down convergence, please consider more epoch if use dropout
    if (dropout):
        total_epoch = total_epoch_dropout
    else:
        total_epoch = total_epoch_no_dropout

    for epoch in range(total_epoch):

        model.train()
        running_loss = 0.0
        running_total = 0
        running_correct = 0
        run_step = 0
        accuracy_train = 0

        for i, (images, labels) in enumerate(train_loader):

            # import images to device (not sure if necessary)
            images = images.to(device)
            labels = labels.to(device)

            # compute the output
            outputs = model(images)
            loss = loss_fn(outputs, labels)

            # reset gradient
            optimizer.zero_grad()

            # compute gradient w.r.t parameters
            loss.backward()

            running_loss += loss.item()
            running_total += labels.size(0)

            # apply corrections
            optimizer.step()

            # get predicted value in training
            with torch.no_grad():
                _, predicted = outputs.max(dim=1)

            running_correct += (predicted == labels).sum().item()
            run_step += 1

            if i % 200 == 0:
                # check accuracy.
                accuracy_train = 100 * running_correct / running_total
                loss_train = running_loss / run_step
                print(f'epoch: {epoch}, steps: {i}, '
                      f'train_loss: {loss_train :.3f}, '
                      f'running_acc: {accuracy_train:.1f} %')
                running_loss = 0.0
                running_total = 0
                running_correct = 0
                run_step = 0

                # If you want to monitor live the performance of the model
                # writer.add_scalar('loss', loss, i)
                # writer.add_scalar('trainacc', accuracy_train, i)

        with torch.no_grad():
            correct_valid = 0
            total = 0
            loss_valid = 0
            run_valid = 0
            model.eval()
            for images, labels in valid_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(dim=1)
                total += labels.size(0)
                correct_valid += (predicted == labels).sum().item()
                loss_valid += (loss_fn(outputs, labels)).item()
                run_valid += 1

            accuracy_validation = 100 * correct_valid / total
            loss_valid = loss_valid / run_valid

            print(f'Validation accuracy: {accuracy_validation} %')
            print(f'Validation loss: {loss_valid :.3f}')
            print(f'Validation error rate: {100 - 100 * correct_valid / total: .2f} %')
            if (accuracy_validation > maxValidAccuracy):
                maxValidAccuracy = accuracy_validation
                epoch_maxValidAccuracy = epoch

            trainingAccuracy.append(accuracy_train)
            trainingLoss.append(loss_train)
            validationAccuracy.append(accuracy_validation)
            validationLoss.append(loss_valid)

            # If you want to monitor live the performance of the model

            # writer.add_scalar('validacc',  accuracy_validation, epoch)
            # writer.flush()

    print(f'Best validation accuracy: {maxValidAccuracy} %')
    print(f'Epoch Best validation accuracy: {epoch_maxValidAccuracy}')

    return np.array(trainingAccuracy), np.array(validationAccuracy), np.array(trainingLoss), np.array(validationLoss)

'''
Function used to test the model
    input: model
    output: test accuracy (scalar)
'''
def test(model):
    with torch.no_grad():
        correct = 0
        total = 0

        #evaluation model

        model.eval()
        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)

            _, predicted = outputs.max(dim=1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        test_acc = 100 * correct / total


    return test_acc

if __name__ == '__main__':

    # Es 1.1 Visualize some images from the training dataset, please choose the number of images you would like to see by
    # changing the value of the variable imagesToPrint
    '''
    imagesToPrint = 3
    for i in range(imagesToPrint):
        block = i==(imagesToPrint-1)
        id = random.randint(0, len(train_set))
        showImg(train_set[id][0], block=block, id=i)

    print('Lenght Train set: ', len(train_set), '\n')
    print('Lenght Test set: ', len(test_set), '\n')
    '''

    #define the model without dropout (es.1.3.1)
    model_no_dropout = convNet(input_size, output_size, apply_dropout=False)
    model_no_dropout.to(device)

    # define the model with dropout (es.1.3.5)
    model_with_dropout = convNet(input_size, output_size, apply_dropout=True)
    model_with_dropout.to(device)


    print('\n---Training model without dropout---\n')
    train_m_no_dropout, valid_m_no_dropout, train_m_no_dropout_loss, valid_m_no_dropout_loss = train_and_validate(model_no_dropout, dropout=False)

    print('\n---Training model with dropout---\n')
    train_m_dropout, valid_m_dropout, train_m_dropout_loss, valid_m_dropout_loss = train_and_validate(model_with_dropout, dropout=True)


    print('\n --- Testing the models --- \n')

    '''Es 1.3.7: Testing the model'''

    test_acc_dropout = test(model_with_dropout)

    print(f'Test accuracy with Dropout: {test_acc_dropout} %')

    '''Es 1.3.4: plot training and validation accuracy'''
    epochs_vec = np.linspace(0, total_epoch_no_dropout, total_epoch_no_dropout)
    plt.figure(0)
    plt.plot(epochs_vec, train_m_no_dropout)
    plt.plot(epochs_vec, valid_m_no_dropout)
    plt.title('Training and Validation Accuracy (no Dropout)')
    plt.xlabel('# Epoch')
    plt.ylabel('Accuracy')
    plt.legend(["Training", "Validation"])
    plt.show(block=False)

    '''Es 1.3.4: plot training and validation losses - No dropout'''
    epochs_vec = np.linspace(0, total_epoch_no_dropout, total_epoch_no_dropout)
    plt.figure(1)
    plt.plot(epochs_vec, train_m_no_dropout_loss)
    plt.plot(epochs_vec, valid_m_no_dropout_loss)
    plt.title('Training and Validation Losses (no Dropout)')
    plt.xlabel('# Epoch')
    plt.ylabel('Loss')
    plt.legend(["Training", "Validation"])
    plt.show(block=False)

    '''Es 1.3.5: compare models with and without dropout'''
    epochs_vec = np.linspace(0, total_epoch_dropout, total_epoch_dropout)
    plt.figure(2)
    plt.plot(epochs_vec, valid_m_dropout)
    plt.plot(epochs_vec[0:total_epoch_no_dropout], valid_m_no_dropout)
    plt.title('Dropout vs No Dropout Validation Accuracy')
    plt.xlabel('# Epoch')
    plt.ylabel('Accuracy')
    plt.legend(["Dropout", "No Dropout"])
    plt.show(block=False)

    '''Es 1.3.5: plot training and validation losses - Dropout'''
    epochs_vec = np.linspace(0, total_epoch_dropout, total_epoch_dropout)
    plt.figure(3)
    plt.plot(epochs_vec, train_m_dropout_loss)
    plt.plot(epochs_vec, valid_m_dropout_loss)
    plt.title('Training and Validation Losses (Dropout)')
    plt.xlabel('# Epoch')
    plt.ylabel('Loss')
    plt.legend(["Training", "Validation"])
    plt.show(block=False)

    '''Es 1.3.5: plot training and validation accuracy - Dropout'''
    epochs_vec = np.linspace(0, total_epoch_dropout, total_epoch_dropout)
    plt.figure(4)
    plt.plot(epochs_vec, train_m_dropout)
    plt.plot(epochs_vec, valid_m_dropout)
    plt.title('Training and Validation Accuracy (Dropout)')
    plt.xlabel('# Epoch')
    plt.ylabel('Accuracy')
    plt.legend(["Training", "Validation"])
    plt.show(block=True)
