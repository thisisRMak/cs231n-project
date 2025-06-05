import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torch


class BaseImageDataset(Dataset):
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.image_files = [
            f
            for f in sorted(os.listdir(self.folder_path))
            if f.endswith((".png", ".jpg", ".jpeg"))
        ]
        self.image_paths = [
            os.path.join(self.folder_path, filename) for filename in self.image_files
        ]

        self.images = [
            Image.open(img_path).convert("RGB") for img_path in self.image_paths
        ]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # filename = self.image_files[idx]
        # img_path = os.path.join(self.folder_path, filename)
        # image = Image.open(img_path).convert('RGB')
        # return image
        return self.images[idx]


class ImageDataset(BaseImageDataset):
    def __init__(self, folder_path):
        path = f"{folder_path}/img"
        super().__init__(path)


class HumanMaskDataset(BaseImageDataset):
    def __init__(self, folder_path):
        path = f"{folder_path}/masks_human"
        super().__init__(path)


class MachineMaskDataset(BaseImageDataset):
    def __init__(self, folder_path):
        path = f"{folder_path}/masks_machine"
        super().__init__(path)


class PetroDataset(Dataset):
    def __init__(self, folder_path):
        print(f"__init__ PetroDataset with: {folder_path}")
        self.path = folder_path
        self.img_dataset = ImageDataset(folder_path)
        self.masks_human_dataset = HumanMaskDataset(folder_path)
        self.masks_machine_dataset = MachineMaskDataset(folder_path)

        assert len(self.img_dataset) == len(self.masks_human_dataset)
        assert len(self.img_dataset) == len(self.masks_machine_dataset)

    def __len__(self):
        return len(self.img_dataset)

    def __getitem__(self, idx):
        img = self.img_dataset[idx]
        mask_human = self.masks_human_dataset[idx]
        mask_machine = self.masks_machine_dataset[idx]

        img_filename = self.img_dataset.image_files[idx]
        mask_human_filename = self.masks_human_dataset.image_files[idx]
        mask_machine_filename = self.masks_machine_dataset.image_files[idx]

        return (
            img,
            mask_human,
            mask_machine,
            img_filename,
            mask_human_filename,
            mask_machine_filename,
        )


#########################################################


def get_subimages(img, sub_image_size):

    image_nparray = np.asarray(img)
    H, W = image_nparray.shape[:2]
    num_subimages_h = H // sub_image_size
    num_subimages_w = W // sub_image_size

    subimages = [
        image_nparray[
            i * sub_image_size: (i + 1) * sub_image_size,
            j * sub_image_size: (j + 1) * sub_image_size,
        ]
        for i in range(num_subimages_h)
        for j in range(num_subimages_w)
    ]
    return subimages


#########################################################

default_sub_image_size = 480

class BaseSubImageDataset(Dataset):
    def __init__(self,
                 folder_path,
                 image_indices=None,
                 sub_image_size=default_sub_image_size,
                 mask=False):
        print(f"__init__ BaseSubImageDataset with: {folder_path}")
        print(f"         , image_indices={image_indices}")
        print(f"         , sub_image_size={sub_image_size}")
        print(f"         , mask={mask}")

        self.folder_path = folder_path
        self.image_files = [
            f
            for f in sorted(os.listdir(self.folder_path))
            if f.endswith((".png", ".jpg", ".jpeg"))
        ]

        if image_indices is not None:
            self.image_files = [
                self.image_files[i]
                for i in image_indices
            ]

        image_paths = [
            os.path.join(self.folder_path, filename)
            for filename in self.image_files
        ]

        self.images = [
            Image.open(img_path).convert("RGB") for img_path in image_paths
        ]

        self.sub_images = []

        for img in self.images:
            subimages = get_subimages(img, sub_image_size)
            if mask:
                subimages = [img[:, :, 0] for img in subimages]

            self.sub_images.extend(subimages)

    def __len__(self):
        return len(self.sub_images)

    def __getitem__(self, idx):
        # filename = self.image_files[idx]
        # img_path = os.path.join(self.folder_path, filename)
        # image = Image.open(img_path).convert('RGB')
        # return image
        return self.sub_images[idx]



# class PetroSubImageDataset(Dataset):
#     def __init__(
#         self,
#         folder_path,
#         image_indices=None,
#         sub_image_size=default_sub_image_size
#     ):
#         print(f"__init__ PetroSubImageDataset: {folder_path}")
#         print(f"         , image_indices={image_indices}")
#         print(f"         , sub_image_size={sub_image_size}")

#         img_folder = f"{folder_path}/img"
#         # masks_human_folder = f"{folder_path}/masks_human"
#         masks_machine_folder = f"{folder_path}/masks_machine"

#         self.img_dataset = BaseSubImageDataset(
#             img_folder, 
#             image_indices = image_indices, 
#             sub_image_size = sub_image_size, 
#             mask=False
#         )

#         self.masks_machine_dataset = BaseSubImageDataset(
#             masks_machine_folder, 
#             image_indices = image_indices, 
#             sub_image_size = sub_image_size, 
#             mask=True
#         )

#         assert len(self.img_dataset) == len(self.masks_machine_dataset)

#     def __len__(self):
#         return len(self.img_dataset)

#     def __getitem__(self, idx):
#         # filename = self.image_files[idx]
#         # img_path = os.path.join(self.folder_path, filename)
#         # image = Image.open(img_path).convert('RGB')
#         # return image
#         return (self.img_dataset[idx], self.masks_machine_dataset[idx])


#########################################################

def permute_and_stack(images, masks):
    # Permute images from [N,H,W,C] to [N,C,H,W]
    images = torch.stack([torch.from_numpy(img) for img in images])
    images = images.permute(0, 3, 1, 2)
    
    # Add channel dimension to masks [N,H,W] -> [N,1,H,W]
    masks = torch.stack([torch.from_numpy(mask) for mask in masks])
    masks = masks.unsqueeze(1)
    
    # Concatenate in channel dimension
    stacked_data = torch.cat([images, masks], dim=1)

    return stacked_data


class PetroSubImageDataset(Dataset):
    def __init__(
        self,
        folder_path,
        image_indices=None,
        sub_image_size=default_sub_image_size
    ):
        print(f"__init__ PetroSubImageDataset: {folder_path}")
        print(f"         , image_indices={image_indices}")
        print(f"         , sub_image_size={sub_image_size}")

        img_folder = f"{folder_path}/img"
        masks_machine_folder = f"{folder_path}/masks_machine"

        img_dataset = BaseSubImageDataset(
            img_folder, 
            image_indices = image_indices, 
            sub_image_size = sub_image_size, 
            mask=False
        )

        masks_machine_dataset = BaseSubImageDataset(
            masks_machine_folder, 
            image_indices = image_indices, 
            sub_image_size = sub_image_size, 
            mask=True
        )

        assert len(img_dataset) == len(masks_machine_dataset)

        self.stacked_data = permute_and_stack(img_dataset.sub_images, masks_machine_dataset.sub_images)


    def __len__(self):
        return len(self.stacked_data)

    def __getitem__(self, idx):

        return self.stacked_data[idx]



class PetroTrainTestSplitDataset(Dataset):
    def __init__(self, folder_path, image_indices=None, sub_image_size=default_sub_image_size):
        self.full_dataset = PetroSubImageDataset(
            folder_path,
            image_indices=image_indices,
            sub_image_size=sub_image_size
        )
        
        # Create train/test indices
        all_indices = list(range(len(self.full_dataset)))
        self.test_indices = [i for i in all_indices if i % 8 in [6, 7]]
        self.train_indices = [i for i in all_indices if i % 8 not in [6, 7]]
        
        # Create train and test datasets
        self.splits = {
            'train': TrainTestSplit(self.full_dataset, self.train_indices),
            'test': TrainTestSplit(self.full_dataset, self.test_indices)
        }

    def __getitem__(self, key):
        if key not in ['train', 'test']:
            raise KeyError("Only 'train' and 'test' splits are available")
        return self.splits[key]

class TrainTestSplit(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]



# Example usae of PetroTrainTestSplitDataset
if __name__ == "__main__":
    
    dataset = PetroTrainTestSplitDataset(folder_path="./dataset/Taskent")

    train_dataset = dataset['train']
    test_dataset = dataset['test']

    
    train_sample = train_dataset[0]
    test_sample = test_dataset[0]

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

