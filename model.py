import torch.nn as nn

class DogCatClassifier(nn.Module):
    def __init__ (self):
        super().__init__()

        # 3 color (RGB) image, so tensor is of shape (B x 3 x H X W)
        # if we take a look, we can see that the images are of size 32 * 32 if we look at them in a file explorer, so our H and W are 32 in this case
        self.conv1 = nn.Conv2d(3, 32, 3, padding = 1) # passes conv kernel over batch and increases num channels from 3 (for RBG) to 32
        self.relu = nn.ReLU(inplace = True) # relu to add nonlinearity
        self.mp = nn.MaxPool2d(2) # reduces h and w of img by a factor of 2
        self.bn1 = nn.BatchNorm2d(32) #normalizes over z distribution https://arxiv.org/abs/1502.03167
        
         # tensor size is now (B x 32 x 32/2 = 16 x 32/2 = 16)

        self.conv2 = nn.Conv2d(32, 64, 3, padding = 1) # 32 channels to 64 ch with 3x3 kernel
        self.bn2 = nn.BatchNorm2d(64) #normalizes over z distribution https://arxiv.org/abs/1502.03167

         # tensor size is now (B x 64 x 32/4 = 8 x 32/4 = 8)

        self.conv3 = nn.Conv2d(64, 128, 3, padding = 1)  # 64 channels to 128 ch with 3x3 kernelnn
        self.bn3= nn.BatchNorm2d(128)  # normalizes over z distribution https://arxiv.org/abs/1502.03167
         
        # tensor size is now (B x 128 x 32/8 = 4 x 32/8 = 4)
        # basically, we have B batches, 128 channels, and a 4x4 pixel representation of our initial image
        self.fc1 = nn.Linear(128 * 4 * 4 , 512)# 2048 feats (ch x h x w), lowkey had to calculator it lol
        self.dropout = 0.5 # tunable, removes half of the values and replaces them with 0s
        self.fc2 = nn.Linear(512, 1) # 512 feats  to 1 scalar output
  
    def forward(self, x):
        x = self.bn1(self.mp(self.relu(self.conv1(x))))
        x = self.bn2(self.mp(self.relu(self.conv2(x))))
        x = self.bn3(self.mp(self.relu(self.conv3(x))))
        x = x.view(x.size(0), -1) # see model_ez for why we do this before linear layers
        # reformats for use in linear layer
        x = self.fc1(x)
        x = nn.functional.relu(x) # relu to add nonlinearity
        x = self.fc2(x)
        x = nn.functional.sigmoid(x) # 1 / 1 + e ^(-x)
        return x
