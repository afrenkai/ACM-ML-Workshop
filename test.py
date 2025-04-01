import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dogs_cats_ds import DogCatDataset
from model import DogCatClassifier
from consts import TEST_DATA







def test(model: nn.Module, test_loader: DataLoader, criterion, device):
    model.eval() # this mode will disable the backward funcitonality for all tensors, and only perform the fwd pass
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad(): # do not update gradients
        for img, lab in test_loader:
            img, lab = img.to(device), lab.to(device).float().view(-1, 1) # similar to how we did it in train, offset both to a gpu for better perf
            # make prediction here
            # evaluate loss here
             #same as with train, if its < 0.5 return 0 here else 1 here 
            total += lab.size(0) # total is increased by batch size
            # correct only += 1 if the prediction matches the label
        print(f'test loss: {test_loss / len(test_loader):.4f}, test_acc: {100*correct/total:.2f}%')
        model.train()


if __name__ == "__main__":
    print('yo')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    dog_test_dataset = DogCatDataset(TEST_DATA)
    dog_test_loader = DataLoader(dog_test_dataset, batch_size = 32, shuffle = False) # since its test, bad to shuffle
    model = DogCatClassifier() # black box for now
    criterion = nn.BCELoss() # loss function
    model.load_state_dict(torch.load('dog_cat_classifier.pth', map_location = device, weights_only = True)) # loads what we trained in train.py
    test(model, dog_test_loader, criterion, device)

