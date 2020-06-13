import re
import os


def is_chinese(uchar):
    """判断一个unicode是否是汉字"""
    if uchar >= '\u4e00' and uchar <= '\u9fa5':
        return True
    else:
        return False


def is_number(uchar):
    """判断一个unicode是否是数字"""
    if uchar >= '\u0030' and uchar <= '\u0039':
        return True
    else:
        return False


def is_alphabet(uchar):
    """判断一个unicode是否是英文字母"""
    if (uchar >= '\u0041' and uchar <= '\u005a') or (uchar >= '\u0061' and uchar <= '\u007a'):
        return True
    else:
        return False


def is_other(uchar):
    """判断是否是（汉字，数字和英文字符之外的）其他字符"""
    if not (is_chinese(uchar) or is_number(uchar) or is_alphabet(uchar)):
        return True
    else:
        return False


def B2Q(uchar):
    """半角转全角"""
    inside_code = ord(uchar)
    if inside_code < 0x0020 or inside_code > 0x7e:  # 不是半角字符就返回原来的字符
        return uchar
    if inside_code == 0x0020:  # 除了空格其他的全角半角的公式为:半角=全角-0xfee0
        inside_code = 0x3000
    else:
        inside_code += 0xfee0
    return chr(inside_code)


def Q2B(uchar):
    """全角转半角"""
    inside_code = ord(uchar)
    if inside_code == 0x3000:
        inside_code = 0x0020
    else:
        inside_code -= 0xfee0
    if inside_code < 0x0020 or inside_code > 0x7e:  # 转完之后不是半角字符返回原来的字符
        return uchar
    return chr(inside_code)


def stringQ2B(ustring):
    """把字符串全角转半角"""
    return "".join([Q2B(uchar) for uchar in ustring])


def give_char_label(cc):
    if is_chinese(cc):
        return 'ZH'
    if is_number(cc):
        return "NUM"
    if is_alphabet(cc):
        return "EN"
    if is_other(cc):
        return "OTHER"


def is_all_chinese(ss):
    return all([is_chinese(i) for i in ss])


def is_all_alphabet(ss):
    return all([is_alphabet(i) for i in ss])


def is_all_number(ss):
    return all([is_number(i) for i in ss])


def is_all_other(ss):
    return all([is_other(i) for i in ss])


def give_label(ss):
    if is_all_chinese(ss):
        return "ZH"
    if is_all_alphabet(ss):
        return "EN"
    if is_all_number(ss):
        return "NUM"
    if is_all_other(ss):
        return "OTHER"
    return "MIXED"
