from torch.utils.data import Dataset
class DogCatDataset(Dataset):
    def __init__(self, ds:Dataset, dog=[5], cat = [3]):
        self.ds = ds
        self.idx = []
        for i in range(len(ds)):
            img, lab = ds[i]
            if lab in dog or lab in cat:
                self.idx.append(i)

    def __len__(self):
        return len(self.idx)
    
    def __getitem__(self, idx):
        orig_idx = self.idx[idx]
        img, lab = self.ds[orig_idx]
        if lab == 5:
            bin_lab = 1
        elif lab == 3:
            bin_lab = 0
        else:
            print('we got a non dog or cat label')

        return img, bin_lab


