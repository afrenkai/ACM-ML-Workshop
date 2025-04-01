import torchvision
from consts import TRAIN_DATA
import torchvision.transforms as transforms
from dogs_cats_ds import DogCatDataset
import random
#optional transformations:
# https://pytorch.org/vision/0.11/transforms.html

#training data using torchvision cifar.


#example of cifar data sample. It is an image, class example.
# here, the image is the image (PIL, or pillow) and the corresponding label. I've chopped the dataset to only include cats
# and dogs, so we can apply a different form of classification so 
r = random.randint(1, len(DogCatDataset(TRAIN_DATA)))
# print(r)
example_data = DogCatDataset(TRAIN_DATA)[r]

print(f'items in an instance of the dataset: {len(example_data)}')
print(f'class corresponding to image: {example_data[1]}')
sample_img = torchvision.transforms.functional.to_pil_image(example_data[0])
sample_img = sample_img.resize((224,224))
sample_img.show()







