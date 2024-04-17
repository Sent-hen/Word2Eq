import csv
from typing import Union
from torch.utils.data import Dataset, DataLoader

class InitialDataset():

    def __init__(self, 
                 datapath,
                 fieldnames):
        self.datapath = datapath
        self.fieldnames = fieldnames
        self.dataset = self.initial_read()
        self.dataset = self.dataset[1:]
        
    def fill_corresponding_numbers(self):
        pass

    def initial_read(self) -> csv.DictReader:
        with open(self.datapath, newline='') as csvfile:
            reader = csv.DictReader(csvfile, fieldnames=fieldnames)
            data = [row for row in reader]
        return data

class TorchDataset(Dataset):

    def __init__(self):
        pass

if __name__ == '__main__':
    datapath = '/Users/cesargamez/Documents/ml-projects/Word2Eq/data/cv_svamp_augmented/fold1/train.csv'
    fieldnames=['Question', 'Numbers', 'Equation', 'Answer', 'group_nums', 'Body', 'Ques']


    dataset = InitialDataset(datapath, fieldnames)
