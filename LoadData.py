import os
from PIL import Image
from torchvision.utils import save_image
import torchvision.transforms as transforms
from torch.utils.data import Dataset


# Load Histology Data from the local directory #
class LoadHistologyData(Dataset):
    def __init__(self, root_img, transform=None, train_flag=False):
        self.root_img = root_img
        self.transform = transform
        self.histology_data = os.listdir(root_img)

    def __len__(self):
        return len(self.histology_data)

    def __getitem__(self, index):
        histology_img = self.histology_data[index]
        histology_path = os.path.join(self.root_img, histology_img)
        histology_image = Image.open(histology_path).convert("RGB")
        if self.transform:
            histology_image = self.transform(histology_image)
        #return histology_image, histology_img
	if train_flag:
        	return histology_image
	else:
		return histology_image, histology_img


# Transformation for Data Augmentation #
def transformation_list_training(new_size=256, height=256, width=256):
    transform_list = transforms.Compose(
        [
            transforms.Resize(new_size),
            #transforms.RandomRotation(degrees=45),
            transforms.RandomHorizontalFlip(), # with comparatively high probability
            transforms.RandomCrop((height, width)),
            #transforms.RandomVerticalFlip(p=0.05), # with low probability
            transforms.ToTensor(),
        ]
    )
    return transform_list
    
    
def transformation_list_no_train(new_size=256, height=256, width=256):
    transform_list = transforms.Compose(
        [
            transforms.Resize(new_size),
            transforms.RandomCrop((height, width)),
            transforms.ToTensor(),
        ]
    )
    return transform_list


# Main driver function #
def test_loaddata():
    transform_list_training = transformation_list_training(new_size=256, height=256, width=256)
    histology_data = LoadHistologyData(root_img='ColNorm/train', transform=transform_list_training)
    print('Data Size: ', len(histology_data))
    example = histology_data[100]
    save_image(example, f"example.ppm")


if __name__ == "__main__":
    test_loaddata()








