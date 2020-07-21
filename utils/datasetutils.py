"""
数据集拆分
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from typing import List
from utils.preprocessing import TextPreHandler, DefaultTextSpliter, Q2BTextReplacer, ZhStopWordRemover
from collections import defaultdict


def split_df(df: pd.DataFrame, test_size=0.3, random_state=30) -> (pd.DataFrame, pd.DataFrame):
    """
    数据集拆分
    :param df: dataFrame text列和label列
    :param test_size:    测试集大小
    :param random_state: 随机种子
    :return: 训练集, 测试集
    """
    df = shuffle(df, random_state=random_state)
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    return train_df, test_df


def split(texts: List[str], labels: List, test_size=0.3, random_state=30) -> (List[str], List[str], List, List):
    """
    数据集拆分
    :param texts: 文本
    :param labels: 标签
    :param test_size: 测试集大小
    :param random_state: 随机种子
    :return: 训练文本, 测试文本, 训练标签, 测试标签
    """
    shuffled_texts, shuffled_labels = shuffle(texts, labels, random_state=random_state)
    train_texts, test_texts, train_labels, test_labels = train_test_split(shuffled_texts, shuffled_labels,
                                                                          test_size=test_size,
                                                                          random_state=random_state)
    return train_texts, test_texts, train_labels, test_labels


def gen_vocab(texts, mindf=10):
    """
    根据数据集生成词表
    :param texts: 数据集
    :param mindf: 最小词频
    :return:
    """
    handler = TextPreHandler(spliter=DefaultTextSpliter(),
                             replacers=[Q2BTextReplacer()],
                             removers=[ZhStopWordRemover()])
    word_lists = handler.multi_prehandle(texts)
    count_dict = defaultdict(int)
    for word_list in word_lists:
        for word in word_list:
            count_dict[word] += 1
    vocab = []
    for word in count_dict:
        if count_dict[word] >= mindf:
            vocab.append(word)
    return vocab


if __name__ == '__main__':
    texts = ['师傅很好，服务很热情！', '送餐员很热情，很好吃谢谢']
    print(gen_vocab(texts, mindf=1))
