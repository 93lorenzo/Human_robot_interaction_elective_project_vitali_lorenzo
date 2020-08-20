from __future__ import print_function

import torch.utils.data as data
from PIL import Image
import os
import os.path
import time

class ImageDataset(data.Dataset):


    def __init__(self, split_filename='train_1.txt', transform=None):

       self.transform = transform

        self.split_filename = split_filename

        self.data = []
        self.labels = []
        self.data_paths = []

        print("filename = "+str(split_filename) )
        #time.sleep(5)
        with open(self.split_filename, 'rt') as f:
            for line in f:
                parts = line.strip().split(' ')
                img_filename = parts[0].strip()
                #print("img filename: " + img_filename)
                full_img_path = img_filename
                if os.path.isfile(full_img_path):
                    self.data.append(full_img_path)
                    self.data_paths.append(parts[0])
                    print("--- parts = " + str(parts) + " ---")

                    self.labels.append(int(parts[1].strip()))
                else:
                    print ("WARNING image ", full_img_path," doesn't exist")

        print("LABELS DENTRO IMAGE DATASET")
        print(set( self.labels) )
        return

    # Loads image from file and returns BGR
    # Note the torchivision transformers work only with PIL images !!
    def img_loader(self, path, cv=False, RGB=False):

        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, 'rb') as f:
            with Image.open(f) as img:
                img_out = img.convert('RGB')

        return img_out

    def get_index(self, element):
        index = self.data_paths.index(element)
        return index

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        sample = self.img_loader(self.data[index])
        target = self.labels[index]

        # ADDED TO OBTAIN THE PATH STRING
        current_path_string = self.data_paths[index]

        if self.transform is not None:
            sample = self.transform(sample)


        # ADDED TO OBTAIN THE PATH STRING
        return sample, target, current_path_string



    def __len__(self):
        return len(self.data)
