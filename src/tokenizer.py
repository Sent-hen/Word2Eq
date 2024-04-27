from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer

import dataset as ds

if __name__ == '__main__':
    tokenizer = Tokenizer(BPE())
    tokenizer.pre_tokenizer = Whitespace()
    dataset = ds.format_dataset()

    trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
    tokenizer.train_from_iterator(dataset.data['Body-Question'], trainer=trainer)
    output = tokenizer.encode("John has 5 apples.")