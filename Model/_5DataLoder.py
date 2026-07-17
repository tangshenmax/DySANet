import os
from glob import glob
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import torch
from torch.utils.data import Dataset, DataLoader


def get_transforms(mode="train", img_size=(224,224)):
    if mode == "train":
        
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=30, p=0.5),
            #A.GaussNoise(std_range=(0.1, 0.25), p=0.5), 
            A.Affine(scale=(0.9,1.1), translate_percent=0.1, rotate=15, p=0.5),
            #A.ElasticTransform(alpha=1, sigma=50, p=0.4), 
            

            #A.Resize(*img_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
        

        '''
        return A.Compose([
            A.Rotate(limit=(-15,15), p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomScale(scale_limit=(-0.15,0.15), p=0.5), 
            #A.Sharpen(alpha=(0.1, 0.3), lightness=(0.9, 1.1), p=0.5), 
            A.RandomBrightnessContrast(brightness_limit=(-0.15,0.15), contrast_limit=(-0.1,0.1), p=0.5),
            #A.GridDistortion(num_steps=5, distort_limit=(-0.3, 0.3), p=0.3), 
            #A.OpticalDistortion(distort_limit=(-0.05, 0.05), p=0.3), 
            #A.HueSaturationValue(p=0.3),
            A.GaussianBlur(blur_limit=(3, 5), sigma_limit=(0.5, 1.5), p=0.5),
            A.GaussNoise(std_range=(0.05, 0.1), p=0.5), 
            #A.ElasticTransform(alpha=0.8, sigma=8, p=0.5),
            #A.CLAHE(clip_limit=2,p=0.3),
            A.Affine(scale=(0.9,1.1), translate_percent=0.15, rotate=15, p=0.5),

            A.Resize(*img_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
        '''
        
    else:
        return A.Compose([
            #A.Resize(*img_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
            ])


class SegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):

        allowed_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

        imgs = []
        for p in sorted(glob(os.path.join(images_dir, "*"))):
            ext = os.path.splitext(p)[1].lower()
            if not os.path.basename(p).startswith('.') and ext in allowed_exts:
                imgs.append(p)
        masks = []
        for p in sorted(glob(os.path.join(masks_dir, "*"))):
            ext = os.path.splitext(p)[1].lower()
            if not os.path.basename(p).startswith('.') and ext in allowed_exts:
                masks.append(p)

        assert len(imgs) == len(masks), \
            f"Number of images and masks differ: {len(imgs)} vs {len(masks)}"
        self.images_paths = imgs
        self.masks_paths = masks
        self.transform = transform

    def __len__(self):
        return len(self.images_paths)

    def __getitem__(self, idx):
        img_path = self.images_paths[idx]
        mask_path = self.masks_paths[idx]

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype('uint8')  

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        else:
            image = ToTensorV2()(image=image)['image']
            mask = torch.from_numpy(mask).long()

        return image, mask


def get_dataloaders(
    data_dir,
    batch_size,
    img_size=(224,224),
    pin_memory=True
):
    
    train_images = os.path.join(data_dir, 'train', 'images')
    train_masks = os.path.join(data_dir, 'train', 'masks')
    val_images = os.path.join(data_dir, 'val', 'images')
    val_masks = os.path.join(data_dir, 'val', 'masks')
    test_images = os.path.join(data_dir, 'test', 'images')
    test_masks = os.path.join(data_dir, 'test', 'masks')

    train_ds = SegmentationDataset(
        train_images,
        train_masks,
        transform=get_transforms('train', img_size)
    )
    val_ds = SegmentationDataset(
        val_images,
        val_masks,
        transform=get_transforms('val', img_size)
    )
    test_ds = SegmentationDataset(
        test_images,
        test_masks,
        transform=get_transforms('test', img_size)
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=pin_memory,
        num_workers = 4
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        pin_memory=pin_memory,
        num_workers = 4
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        pin_memory=pin_memory,
        num_workers = 4
    )

    return train_loader, val_loader, test_loader
