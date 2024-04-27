import csv
import json
from torch.utils.data import Dataset

def format_dataset(datapath='/Users/cesargamez/Documents/ml-projects/Word2Eq/data/SVAMP.json'):
    return SVAMPDataset(datapath, [['Body', 'Question']], ['ID', 'Body', 'Question', 'Answer', 'Type'])


class SVAMPDataset:
    def __init__(self, datapath, combine=None, removables=None):
        self.datapath = datapath
        self.data = self.initial_read()
        self.combine = combine
        self.removables = removables
        self.combine_cols()
        self.remove_cols()
        self.data = self.to_dict() # finial type of self.data is dict
            
    def to_dict(self):
        data = dict()
        for group in self.data:
                for k in group:
                    if(k not in data):
                        data[k] = []
                    data[k].append(group[k])
        return data
    
    def initial_read(self) -> list:
        with open(self.datapath) as jsonfile:
            data = json.load(jsonfile)
        return data
    
    def combine_cols(self):
        if(self.combine is not None):
            for group in self.data:
                for pair in self.combine:
                    comb1 = pair[0]
                    comb2 = pair[1]
                    group[f'{comb1}-{comb2}'] = (group[comb1] + ' ' + group[comb2] if group[comb1][-1] == '.' else group[comb1] + ', ' + group[comb2])
    
    def remove_cols(self):
        if(self.removables is not None):
            for group in self.data:
                for col in self.removables:
                    del group[col]

    def individualize_col(self, col):
        for group in self.data:
            return group[col]