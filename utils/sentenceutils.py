import jieba
import sys, os
import utils.wordutils as wordutils
sys.path.append("")

def sentence2charlist(sentence):
    """

    :param sentence: sentence
    :return:
    """
    res_list = []
    for ch in sentence:
        if wordutils.is_chinese(ch):
            res_list.append(ch)
        elif wordutils.is_number(ch):
            res_list.append('NUM')
        elif not wordutils.is_other(ch):
            res_list.append(ch)
    return ' '.join(res_list)


def sentence2wordlist(sentence):
    """

    :param sentence: sentence
    :return:
    """
    res_list = []
    for word in jieba.cut(sentence):
        word = word.strip('\n')
        if wordutils.is_all_chinese(word):
            res_list.append(word)
        elif wordutils.is_all_number(word):
            res_list.append('NUM%d' % len(word))
        elif not wordutils.is_all_other(word):
            res_list.append(word)
    return ' '.join(res_list)
