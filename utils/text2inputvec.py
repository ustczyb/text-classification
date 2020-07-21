"""
将word list 转换为输入到模型中的feature(vector)
"""
import sys
from abc import ABCMeta, abstractmethod
from typing import List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.append("")


class AbstractVectorizer(metaclass=ABCMeta):
    """

    """

    @abstractmethod
    def to_vector(self, word_list):
        pass

    @abstractmethod
    def multi_to_vector(self, multi_word_list: List[List[str]]):
        pass


class LabelVectorizer():
    def __init__(self, filename):
        self._label_index_dict = {}
        with open(filename, 'r') as f:
            label_list = f.readlines()
        for i, label in enumerate(label_list):
            label = label.replace("\n", "")
            self._label_index_dict[label] = i

    def labels_to_id(self, labels):
        label_list = labels.split(',')
        label_index_list = [1 if label in label_list else 0 for label in self._label_index_dict]
        return label_index_list


class TfidfVectorizerProxy(AbstractVectorizer):
    """
    tf_idf vectorizer
    """

    def __init__(self, min_df=5, ngram=1):
        self.vectorizer = TfidfVectorizer(min_df=min_df, ngram_range=(1, ngram))

    def fit(self, multi_token_list: List[List[str]]) -> None:
        sentence_list = list(map(lambda l: ' '.join(l), multi_token_list))
        self.vectorizer.fit(sentence_list)

    def to_vector(self, word_list: List[str]) -> np.array:
        text_with_space = ' '.join(word_list)
        return self.vectorizer.transform([text_with_space]).toarray()

    def multi_to_vector(self, multi_token_list: List[List[str]]) -> np.array:
        text_list = list(map(lambda l: ' '.join(l), multi_token_list))
        return self.vectorizer.transform(text_list)


class WordVecVectorizer(AbstractVectorizer):
    """
    词向量生成句向量
    """

    def __init__(self, wordvec_file):
        pass

    def to_vector(self, word_list):
        pass

    def multi_to_vector(self, multi_word_list: List[List[str]]):
        pass


class IndexVectorizer(AbstractVectorizer):
    """
    to index
    """

    def __init__(self, max_len, dict_file='../00_data/vocab/ZH-char.txt'):
        self._max_len = max_len
        self._word_index_dict = {}
        self._UNK = 1
        self._PAD = 0
        self._read_dict(dict_file)

    def _read_dict(self, filename):
        with open(filename, 'r') as f:
            char_list = f.readlines()
        for i, char in enumerate(char_list):
            char = char.replace("\n", "")
            self._word_index_dict[char] = i
        print("vocabulary of char generated....")

    def to_vector(self, word_list):
        index_list = [self._word_index_dict.get(x, self._UNK) for x in word_list if x.strip()]
        text_len = len(index_list)
        if text_len > self._max_len:
            return index_list[:self._max_len]
        else:
            pad_length = self._max_len - text_len
            return index_list + [self._PAD] * pad_length

    def multi_to_vector(self, multi_word_list: List[List[str]]):
        return [self.to_vector(x) for x in multi_word_list]

if __name__ == '__main__':
    vectorizer = IndexVectorizer(max_len=80)
    print(vectorizer.to_vector(['师', '傅', '很', '好', '服', '务', '热', '情', '!', '送', '餐', '员', '吃', '谢']))
