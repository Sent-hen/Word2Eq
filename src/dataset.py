import csv
import json
from torch.utils.data import Dataset


class SVAMPDataset:

    def __init__(self, datapath):
        self.datapath = datapath
        self.data = self.initial_read()

    def initial_read(self) -> list:
        with open(self.datapath) as jsonfile:
            data = json.load(jsonfile)
        return data


class InitialDataset:

    def __init__(self, datapath, fieldnames):
        self.datapath = datapath
        self.fieldnames = fieldnames
        self.dataset = self.initial_read()
        self.dataset = self.dataset[1:]
        
    def fill_corresponding_numbers(self):
        pass

    def initial_read(self) -> csv.DictReader:
        with open(self.datapath, newline='') as csvfile:
            reader = csv.DictReader(csvfile, fieldnames=self.fieldnames)
            data = [row for row in reader]
        return data

class TorchDataset(Dataset):

    def __init__(self):
        pass


if __name__ == '__main__':
    datapath = '../data/SVAMP.json'

    dataset = SVAMPDataset(datapath)

    # dataset.data = dataset.data[:4]

    for group in dataset.data:
        group["Body_with_Question"] = (group["Body"] + ' ' + group["Question"] if group["Body"][-1] == '.' else group["Body"] + ', ' + group["Question"])
        del group["ID"]
        del group["Body"]
        del group["Question"]

    with open('SVAMP_combined_body_question.json', 'w', encoding='utf-8') as f:
        json.dump(dataset.data, f, ensure_ascii=False, indent=4)
