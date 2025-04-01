import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from consts import TRAIN_DATA
from tqdm import tqdm
from model import DogCatClassifier
from dogs_cats_ds import DogCatDataset



# def train_sgd(model: nn.Module,
#               imgs: torch.Tensor,
#               labels: torch.Tensor,
#               batch_size: int,
#               criterion: nn.Module,
#               optimizer: optim.Optimizer):
#     losses: list = []
#     shuffle: torch.Tensor = torch.randperm(imgs.size(0))
#     images_shuffled: torch.Tensor = imgs[shuffle]
#     labels_shuffled: torch.Tensor = labels[shuffle]
#     for i in range(0, imgs.size(0), batch_size):
#         batched_images: torch.Tensor = images_shuffled[i:i+batch_size]
#         batched_labels: torch.Tensor = labels_shuffled[i:i+batch_size]
#         outputs = model(batched_images)
#         loss = criterion(outputs, batched_labels)
#         loss.backward()
#         optimizer.step()
#
#         losses.append(loss.item())
#     return losses



def train(model: nn.Module, train_loader: DataLoader, criterion, optimizer, device, epochs):
    model.to(device) # send to gpu if there is one, otherwise toss it over to cpu 
    model.train() #train mode means that all gradients are active and modifiable

    for epoch in tqdm(range(epochs)): # wrapper around for loop to add a nice progress bar
        running_loss = 0.0 # start the loss, amount of cats and dogs we guess correctly, and complete samples at 0 (float 0 in case of loss since it can be a float)
        correct = 0
        total = 0
        for i, (img, lab) in enumerate(train_loader): # for each image, label pair in the dataset 
            img, lab = img.to(device), lab.to(device).float().view(-1, 1) # send the image and label to the gpu if there is one else send to cpu, .view(-1, 1) returns the same tensor data but with the shape of the last dimension
            optimizer.zero_grad() # resets gradients to zero when we initialize.

            # get a prediction here
                    #
            # calculate the loss here
                    
            # perform backpropogation here
            
                  
            #
            optimizer.step() # update optimizer 

            running_loss += loss.item() # loss in epoch updated with loss
            #calculate accuracy


            #hint: what kind of values is the accuracy outputing? What kind of values do we want?
            
            total += lab.size(0) #total samples is increased by the 0th dim of the tensor(batch size)
            # only add 1 to the correct count if the actual label (dog) = the predicted label(dog)

            if (i + 1) % 50 == 0:
                print(f'yo its epoch {epoch + 1} out of {epochs} and we on minibatch {i + 1} / {len(train_loader)}. Loss lookin like: {running_loss/(i+1):.4f}, acc lookin like {100 * correct / total :.2f}%')
                running_loss = 0.0
                total = 0
                correct = 0


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    dog_train_dataset = DogCatDataset(TRAIN_DATA)
    dog_train_loader = DataLoader(dog_train_dataset, batch_size = 32, shuffle = True) # since its train, ok to shuffle

    model = DogCatClassifier() # black box for now
    criterion = nn.BCELoss() # cross entropy loss, feel free to experiment with others
    optimizer = optim.Adam(model.parameters(), lr = 0.001) # feel free to mess around with other optimizers as well

    train(model = model, train_loader = dog_train_loader, criterion = criterion, optimizer = optimizer, device = device, epochs = 10)
    torch.save(model.state_dict(), 'dog_cat_classifier.pth') # saves model to pth file, which can be read by pytorch
    print('done w train, model saved')

  
