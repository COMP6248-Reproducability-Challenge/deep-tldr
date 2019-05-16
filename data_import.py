import os

import torch
import torchtext.data as td
import torchtext.data.dataset
import torchtext.vocab
from torchtext import vocab
from torchtext.data import BucketIterator
from torchtext.data import Example

import log
from config import *
# print(f'There are {len(glove.itos)} words in the vocabulary')
from utils.batch_generator import BatchGenerator

logger = log.logger

strings_to_remove = ['<t>', '</t>', '-lrb- cnn -rrb-', '-lrb- reuters -rrb-']
strings_to_replace = {'-lrb-': '(', '-rrb-': ')', '-lcb-': '{', '-rcb-': '}', '--': '-'}
special_chars = ['<sos>', '<eos>', '<unk>', '<pad>']


class DataProcessing:
    """
    Class to process the data.

    train_data, val_data and test_data contain the genorators for tensors for each batch containing each the indecies for each article and its summary. Should also magically contain the word embeddings for use with nn.Embeddings()
    The indices correspond to the vocab in self.text.vocab (for articles) and self.label.vocab (for text). This means they are not the same as in GloVe, HOWEVER, the embeddings should be the same for the purposes of encoding/decoding - just have to be careful to ensure you're getting good values from stoi/itos
    """

    def __init__(self):
        logger.info("Importing GloVe...")
        self.embedding_size = 50
        self.article_size = 800 if not DEV_MODE else 400
        glove = vocab.GloVe(name='6B', dim=self.embedding_size)
        logger.info("Complete")
        tokenize = lambda x: x.split()
        self.text = td.Field(sequential=True, tokenize=tokenize, fix_length=self.article_size, init_token='<sos>',
                             eos_token='<eos>')
        self.label = td.Field(sequential=True, tokenize=tokenize, fix_length=50, init_token='<sos>', eos_token='<eos>')
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.build_datasets(glove)
        print(f'There are {len(self.text.vocab.itos)} words in the vocabulary')
        logger.info("Data import complete.")
        logger.info("\n********************\nUSING POTATO 2 DATASET\n********************") if USE_D2 else None

    # Have to do this now as not trimming lines massively increases memory use
    @staticmethod
    def preprocess_text(line: str, num_tokens: int):
        """
        Preprosesses each article by truncating to correct length and removing/replacing substrings where appropriate

        Use immediately after reading lines to minimise memory use
        :param line: The article
        :param num_tokens: The number of tokens to truncate the article too
        :return: The processed line
        """
        # Do other preprocessing before this line
        for s in strings_to_remove:
            line = line.replace(s, '')
        for key, value in strings_to_replace.items():
            line = line.replace(key, value)
        line = " ".join(line.split(' ', num_tokens))
        return line

    def process_file(self, src_filename: str, tagged_filename: str):
        """
        Converts each pair of src/summary files into a dataset
        :param src_filename:
        :param tagged_filename:
        :return:
        """
        src = open(src_filename, 'r', encoding="utf-8")
        tag = open(tagged_filename, 'r', encoding="utf-8")
        srclines = [self.preprocess_text(x, self.article_size) for x in src]
        taglines = [self.preprocess_text(x, 50) for x in tag]
        if DEV_MODE:
            # default length: 287227
            srclines = srclines[:500]
            taglines = taglines[:500]
        datazip = zip(srclines, taglines)
        examples = [Example.fromdict({'text': text, 'label': label},
                                     {'text': ('text', self.text), 'label': ('label', self.label)}) for (text, label) in
                    datazip]
        dataset = torchtext.data.Dataset(examples=examples, fields={'text': self.text, 'label': self.label})
        return dataset

    def build_datasets(self, glove):
        """
        Builds the datasets and the vocab and makes it iterable. Tensors returned are of size [# tokens in article/ground truth] x 50 (batch size) containing the indices which can be converted to text using 'vocab.itos'
        :return:
        """
        data_path = '/data_fake/' if USE_D2 else '/data/'
        logger.info("Building datasets...")
        train = self.process_file(os.getcwd() + data_path + 'train.txt.src',
                                  os.getcwd() + data_path + 'train.txt.tgt.tagged')
        val = self.process_file(os.getcwd() + data_path + 'val.txt.src', os.getcwd() + data_path + 'val.txt.tgt.tagged')
        test = self.process_file(os.getcwd() + data_path + 'test.txt.src',
                                 os.getcwd() + data_path + 'test.txt.tgt.tagged')
        logger.info("Complete")
        logger.info("Building Vocab...")
        self.text.build_vocab(train.text, val.text, test.text, train.label, val.label, test.label, vectors=glove,
                              max_size=100000000)
        self.label.vocab = self.text.vocab

        logger.info("Complete")
        logger.info("Building iterators...")
        self.train_data = BatchGenerator(BucketIterator(dataset=train, train=True, batch_size=BATCH_SIZE))
        self.val_data = BatchGenerator(BucketIterator(dataset=val, train=False, batch_size=BATCH_SIZE))
        self.test_data = BatchGenerator(BucketIterator(dataset=test, train=False, batch_size=BATCH_SIZE))
        logger.info("Complete")

    @staticmethod
    def data_to_embeddings(data: torch.Tensor, field: td.Field):
        def word_to_embedding(word):
            return field.vocab.vectors[word]

        embeddings = torch.Tensor(data.shape[0], data.shape[1], field.vocab.vectors.shape[1])
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                w = data[i, j].item()
                embeddings[i, j] = word_to_embedding(w)
        return embeddings

    @staticmethod
    def closest_word_index(embedding: torch.Tensor, embedding_vectors: torch.Tensor):
        # tensors can't be Long data type
        distances = [torch.dist(embedding.squeeze().double(), e.double()) for e in embedding_vectors]
        return distances.index(min(distances))


def test_misc():
    dp = DataProcessing()
    print("UNK index in text:" + str(dp.text.vocab.stoi['<unk>']))
    print("UNK index in summaries:" + str(dp.text.vocab.stoi['<unk>']))
    print("PAD index in text:" + str(dp.text.vocab.stoi['<pad>']))
    print("PAD index in summaries:" + str(dp.text.vocab.stoi['<pad>']))
    print("SOS index in text:" + str(dp.text.vocab.stoi['<sos>']))
    print("SOS index in summaries:" + str(dp.text.vocab.stoi['<sos>']))
    print("EOS index in text:" + str(dp.text.vocab.stoi['<eos>']))
    print("EOS index in summaries:" + str(dp.text.vocab.stoi['<eos>']))
    for X, y in dp.train_data:
        print(type(X))
        print(X.shape)
        print(dp.data_to_embeddings(X, dp.text).shape)
        print(y.shape)
        print(' ')
    logger.info("Finished")


if __name__ == '__main__':
    test_misc()
