"""
预处理
把文本处理成list 便于后续的处理
"""
import jieba
import sys
import utils.wordutils as wordutils
from abc import ABCMeta, abstractmethod
from typing import List

sys.path.append("")


class TextReplacer(metaclass=ABCMeta):
    """
    用于进行特殊字符的替换 删除等 例如去掉emoji 将所有数字统一替换成1 小写转大写等
    """

    @abstractmethod
    def handle_text(self, sentence: str) -> str:
        pass


class TextSpliter(metaclass=ABCMeta):
    """
    用于将句子拆分成list 可以是分词 也可以是单字的拆分
    """

    @abstractmethod
    def split_text(self, sentence: str) -> List[str]:
        pass


class StopWordRemover(metaclass=ABCMeta):
    """
    去除停用词 词替换
    """

    @abstractmethod
    def remove_stopwords(self, words: List[str]) -> List[str]:
        pass


class TextPreHandler():
    """
    文本预处理
    """

    def __init__(self, spliter: TextSpliter, replacers: List[TextReplacer] = [], removers: List[StopWordRemover] = []):
        self._text_replacers = replacers
        self._text_spliter = spliter
        self._stopword_removers = removers

    def pre_handle(self, text: str) -> List[str]:
        for replacer in self._text_replacers:
            text = replacer.handle_text(text)
        token_list = self._text_spliter.split_text(text)
        for remover in self._stopword_removers:
            token_list = remover.remove_stopwords(token_list)
        return token_list

    def multi_prehandle(self, texts: List[str]) -> List[List[str]]:
        return list(map(self.pre_handle, texts))


class Q2BTextReplacer(TextReplacer):

    def handle_text(self, text: str) -> str:
        text = text.lower()
        ss = []
        for s in text:
            rstring = ""
            for uchar in s:
                inside_code = ord(uchar)
                if inside_code == 0x3000:  # 全角空格直接转换
                    inside_code = 0x0020
                elif inside_code >= 0xff01 and inside_code <= 0xff5e:  # 全角字符（除空格）根据关系转化
                    inside_code -= 0xfee0
                rstring += chr(inside_code)
            ss.append(rstring)
        return ''.join(ss)


class JiebaTextSpliter(TextSpliter):
    """
    Jieba分词
    """

    def split_text(self, sentence: str) -> List[str]:
        return list(jieba.cut(sentence))


class DefaultTextSpliter(TextSpliter):
    """
    不分词 每个字符作为一个词
    """

    def split_text(self, sentence: str) -> List[str]:
        res_list = []
        for ch in sentence:
            res_list.append(ch)
        return res_list


class ZhStopWordRemover(StopWordRemover):
    """
    中文停用词去除
    """

    def __init__(self, stopword_file='/Users/zyb/code/text-classification/utils/stopwords-master/hit_stopwords.txt'):
        self.stop_words = set()
        with open(stopword_file, 'r') as f:
            words = f.readlines()
        for word in words:
            self.stop_words.add(word.strip('\n'))

    def remove_stopwords(self, words: List[str]) -> List[str]:
        return list(filter(lambda word: word not in self.stop_words, words))


if __name__ == '__main__':
    prehandler = TextPreHandler(spliter=JiebaTextSpliter(),
                                replacers=[Q2BTextReplacer()],
                                removers=[ZhStopWordRemover()])
    word_list = prehandler.multi_prehandle(["很喜欢喝他家的饮料！谢谢送餐小哥大冷天送来！",
                                            "经过上次晚了2小时，这次超级快，20分钟就送到了……"])
    print(word_list)
