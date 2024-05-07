from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer

import dataset as ds


dataset = ds.format_dataset()

class TokenizedData:
    def __init__(self, data):
        self.tokenizer = Tokenizer(BPE())
        self.tokenizer.pre_tokenizer = Whitespace()
        self.trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
        self.data = data

        self.train()
        self.data = self.encode()

    def decdoe(self, input):
        pass

    def encode(self):
        encoded_data = []
        for sample in self.data:
            encoded_data.append(self.tokenizer.encode(sample))
        return encoded_data

    def train(self):
        self.tokenizer.train_from_iterator(self.data, trainer=self.trainer)


if __name__ == '__main__':
    body_question_tokenized_data = TokenizedData(dataset.data['Body-Question'])
    equation_tokenized_data = TokenizedData(dataset.data['Equation'])
    for d in equation_tokenized_data.data:
        print(f'{d.tokens}\n')