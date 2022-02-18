import glob
from math import ceil
from pydoc import classname, splitdoc
import random
import numpy as np
import os
import shutil



random.seed(0)

class Create_Database:
    def __init__(self, src = "Dataset", dst = "images", split = [0.7, 0.2, 0.1]) -> None:
        self.signals = os.listdir(src)
        print(self.signals)
        self.src = src
        self.dst = dst
        self.seed = random.seed(random.random())
        self.splits = [split[0], split[1]/ (split[1]+ split[2]), None]

    def ds_train(self):
        print(self.signals)
        dst_path = os.path.join(self.dst, "train")
        [os.makedirs(os.path.join(dst_path, x), exist_ok=True) for x in self.signals]

        for sig in self.signals:
            path = os.path.join(self.src, sig)
            files = os.listdir(path)
            random.shuffle(files)
            print(len(files))
            train = files[:ceil(self.splits[0]*len(files))]
            print(len(train))

            for file in train:
                file = os.path.join(self.src, sig, file)
                shutil.copy2(file, os.path.join(dst_path, sig))

    def ds_valid(self):
        dst_path = os.path.join(self.dst, "val")
        [os.makedirs(os.path.join(dst_path, x), exist_ok=True) for x in self.signals]

        for sig in self.signals:
            path = os.path.join(self.src, sig)
            files = os.listdir(path)
            random.shuffle(files)
            print(len(files))
            val = files[:ceil(self.splits[1]*len(files))]
            print(len(val))

            for file in val:
                file = os.path.join(self.src, sig, file)
                shutil.copy2(file, os.path.join(dst_path, sig))

    def ds_test(self):
        dst_path = os.path.join(self.dst, "test")
        [os.makedirs(os.path.join(dst_path, x), exist_ok=True) for x in self.signals]

        for sig in self.signals:
            path = os.path.join(self.src, sig)
            files = os.listdir(path)

            for file in files:
                file = os.path.join(self.src, sig, file)
                shutil.copy2(file, os.path.join(dst_path, sig))

    def create(self):
        self.ds_train()
        self.ds_valid()
        self.ds_test()

Create_Database(src="Dataset", dst = "images", split = [0.7, 0.2, 0.1]).create()


