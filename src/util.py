import numpy as np
import torch, os
from torch.utils.data import Dataset
import pandas as pd
import csv, random
from torch.autograd import Variable


def write_pred(test_pred, test_idx, file_path):
    test_pred = [item for sublist in test_pred for item in sublist]
    with open(file_path, "w") as f:
        for idx, pred in zip(test_idx, test_pred):
            print(idx.upper() + "," + str(pred[0]), file=f)


def write_csv(filePath, rate):

    # file_dir=os.listdir(filePath)
    for root, dirs, files in os.walk(filePath):
        file_dir = dirs
        file_list = files
        break
    train = []
    test = []
    for i in range(0, len(file_dir)):
        templist = os.listdir(filePath + "/" + file_dir[i])
        offset = int(len(templist) * rate)
        random.shuffle(templist)
        for j in range(0, len(templist)):
            if j < offset:
                train.append([filePath + "/" + file_dir[i] + "/" + templist[j], str(i)])
            else:
                test.append([filePath + "/" + file_dir[i] + "/" + templist[j], str(i)])

    random.shuffle(train)
    random.shuffle(test)
    with open(filePath + "/train.csv", "w+", newline="") as f:
        # create the csv writer
        writer = csv.writer(f)
        # write a row to the csv file
        for row in train:
            writer.writerow(row)
    with open(filePath + "/val.csv", "w+", newline="") as f:
        # create the csv writer
        writer = csv.writer(f)
        # write a row to the csv file
        for row in test:
            writer.writerow(row)
    print("DATA=-2-=CSV  is OK")


# Dataset preparation
class ExeDataset(Dataset):
    def __init__(self, data_path, first_n_byte=2000000):
        self.data_path = data_path
        self.first_n_byte = first_n_byte
        files = []
        labels = []
        with open(self.data_path, "r") as csv_file:
            csv_reader = csv.reader(csv_file)
            # Passing the cav_reader object to list() to get a list of lists
            list_rows = list(csv_reader)
        for row in list_rows:
            files.append(row[0])
            labels.append(int(row[1]))

        self.files = files
        self.labels = labels

    def read_csv(self):
        with open(self.data_path, "r") as csv_file:
            csv_reader = csv.reader(csv_file)
            # Passing the cav_reader object to list() to get a list of lists
            list_rows = list(csv_reader)
        return list_rows

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        try:
            with open(self.files[idx], "rb") as f:
                tmp = [i + 1 for i in f.read()[: self.first_n_byte]]
                tmp = tmp + [0] * (self.first_n_byte - len(tmp))
                exedata = torch.from_numpy(np.array(tmp))
                tmp = Variable(exedata.long(), requires_grad=False)
        except:
            print("=====================")
        # mmm=np.array([self.labels[idx]])
        labels = torch.from_numpy(np.array([self.labels[idx]]))
        labels = Variable(labels.float(), requires_grad=False)
        return tmp, labels
