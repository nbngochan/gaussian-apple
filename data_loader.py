import os
import cv2
import json
import torch
import numpy as np
from utils.util import smoothing_mask, total_size
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class ImageTransform():
    def __init__(self):
        pass
    
    def __call__(self):
        pass


class AppleDataset(Dataset):
    """
    Surface Defective Apple Dataset
    """
    def __init__(self, mode, data_path, img_size=(512, 512), transform=None):
        self.data_path = data_path
        self.mode = mode
        self.transform = transform
        self.num_classes = 2
        self.img_size = img_size
        self.dataset = self.load_data()
        
        n = len(self.dataset)
        
        if mode == 'train':
            self.dataset = self.dataset[:int(n*0.8)]
            
        elif mode == 'test':
            self.dataset = self.dataset[int(n*0.8):]
        
        
    def __len__(self):
        return len(self.dataset)
    
    
    def __getitem__(self, idx):
        annotation, image_path, image = self.get_annotation(idx)
        sum_size = 1
        height, width = image.size
        
        
        # if self.transform:
        #     image = self.transform(image)
        
        # target_size = self.img_size[0], self.img_size[1]
        mask = np.zeros((self.img_size[0], self.img_size[1], self.num_classes), dtype=np.float32)
        area = np.zeros((self.img_size[0], self.img_size[1], self.num_classes), dtype=np.float32)
        
        target = np.array(annotation)
        boxes = target[:, :8] if target.shape[0]!=0 else None
        labels = target[:, 8] if target.shape[0]!=0 else None
        
        # Apply transform on `image`, `boxes`, `labels`
        # image, boxes, labels = self.transform(image, boxes, labels)
        
        # Recompute the coordinate when image size changes
        if boxes is not None:
            target_h, target_w = self.img_size[0], self.img_size[1]
            new_wh = np.array([target_w / width, target_h / height]) # rescaling factor
            boxes = boxes * np.tile(new_wh, 4)
            
            labels = labels.astype(np.int32)
            
            num_obj = len(boxes) if boxes is not None else 1
            sum_size = total_size(boxes)
            
            for box, label in zip(boxes, labels):
                mask, area = smoothing_mask(mask, area, box, sum_size/num_obj, label)
        

        image, mask, area = self.annotation_transform(np.array(image), mask, area, self.img_size[1], self.img_size[1])
                         
        image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask.astype(np.float32))
        area = torch.from_numpy(area.astype(np.float32))
        sum_size = torch.from_numpy(np.array([sum_size], dtype=np.float32))
           
        return image, mask, area, sum_size
    
    
    def annotation_transform(self, image, mask, area, height, width):
        resize_img = cv2.resize(image, (width, height))
        resized_mask = cv2.resize(mask, (width, height))
        resized_area = cv2.resize(area, (width, height))
        return resize_img, resized_mask, resized_area
    
    def load_data(self):
        # read ground truth json file
        with open(os.path.join(self.data_path, 'ground-truth','ground_truth.json')) as f:
            data = json.load(f)
        return data


    def get_annotation(self, idx):
        sample = self.dataset[idx]
        
        image_path = os.path.join(self.data_path, 'images', sample['name'])
        class_id = int(sample['class'])
        image = Image.open(image_path).convert('RGB')
        width, height = image.size
        
        temp_boxes = sample['crop_coordinates_ratio']
        annotations = []
        
        # convert format from [cx, cy, w, h] -> [x1, y1, x2, y2, x3, y3, x4, y4, class_id]
        for box in temp_boxes:
            cx, cy, w, h = box
            x1 = int((cx - w / 2) * width)
            y1 = int((cy - h / 2) * height)
            x2 = int((cx + w / 2) * width)
            y2 = int((cy - h / 2) * height)
            x3 = int((cx + w / 2) * width)
            y3 = int((cy + h / 2) * height)
            x4 = int((cx - w / 2) * width)
            y4 = int((cy + h / 2) * height)
            
            # perform boundary checks
            x1 = max(0, min(x1, width - 1))
            y1 = max(0, min(y1, height - 1))
            x2 = max(0, min(x2, width - 1))
            y2 = max(0, min(y2, height - 1))
            x3 = max(0, min(x3, width - 1))
            y3 = max(0, min(y3, height - 1))
            x4 = max(0, min(x4, width - 1))
            y4 = max(0, min(y4, height - 1))

            annotations.append([x1, y1, x2, y2, x3, y3, x4, y4, class_id])
                   
        return annotations, image_path, image 
    
    
if __name__ == '__main__':
    appledata = AppleDataset(mode='train',
                             data_path='/root/data/apple/cropped-apple-bb/',
                             img_size=(512, 512))
    
    apple_loader = DataLoader(appledata, batch_size=32, shuffle=True)
    for batch in apple_loader:
        images, masks, areas, total_sizes = batch