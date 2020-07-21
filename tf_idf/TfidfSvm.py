from utils import preprocessing, text2inputvec, datasetutils
import pandas as pd
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
import numpy as np
import sklearn.metrics as metrics


class TfidfClassifier(object):

    def __init__(self, prehandler, min_df=5, ngram=1, C=1, multi_label=False, label_nums=2):
        self.prehandler = prehandler
        self.multi_label = multi_label
        self.vectorizer = text2inputvec.TfidfVectorizerProxy(min_df, ngram)
        if not multi_label:
            self.classifier = SVC(kernel='linear', C=C)
        else:
            self.classifier = OneVsRestClassifier(SVC(kernel='linear', C=C))

    def train(self, train_texts, train_labels) -> None:
        """
        模型训练
        :param train_texts:
        :param train_labels:
        :return:
        """
        # 1. 训练vectorizer
        train_token_list = self.prehandler.multi_prehandle(train_texts)
        self.vectorizer.fit(train_token_list)
        train_vector = self.vectorizer.multi_to_vector(train_token_list)
        # 2.训练分类器
        if not self.multi_label:
            self.classifier.fit(train_vector, train_labels)
        else:
            pass

    def evaluation(self, test_texts, test_labels):
        """
        模型评估
        :param test_texts: 测试数据
        :param test_labels: 测试标签
        :return:
        """
        test_vector = self.vectorizer.multi_to_vector(self.prehandler.multi_prehandle(test_texts))
        predict_labels = self.classifier.predict(test_vector)
        if not self.multi_label:
            acc_score = metrics.accuracy_score(test_labels, predict_labels)
            prec_score = metrics.precision_score(test_labels, predict_labels, average='macro')
            recall_score = metrics.recall_score(test_labels, predict_labels, average='macro')
            confusion_matrix = metrics.confusion_matrix(test_labels, predict_labels)
            print('acc_score: %f,prec_score: %f, recall_score: %f' % (acc_score, prec_score, recall_score))
            print('confusion_matrix:\n', confusion_matrix)
        else:
            pass
        each_res = predict_labels == test_labels
        badcase_text = np.asarray(test_texts)[each_res]
        badcase_true_label = np.asarray(test_labels)[each_res]
        badcase_pred_label = np.asarray(predict_labels)[each_res]
        return badcase_text, badcase_true_label, badcase_pred_label

    def predict_single(self, text: str):
        text_vector = self.vectorizer.to_vector(self.prehandler.pre_handle(text))
        return self.classifier.predict(text_vector)


def load_dataset():
    data_df = pd.read_csv('../00_data/cn_short/waimai_10k.tsv', sep='\t')
    print('数据集大小:' + str(data_df.shape))
    train_df, test_df = datasetutils.split_df(data_df, test_size=0.2)
    return train_df['text'], train_df['label'], test_df['text'], test_df['label']


if __name__ == '__main__':
    # 1.加载数据集
    df = pd.read_csv('../00_data/cn_short/waimai_10k.tsv', sep='\t')
    train_df, test_df = datasetutils.split_df(df, test_size=0.2)
    train_text_list = train_df['text'].values.tolist()
    test_text_list = test_df['text'].values.tolist()
    train_labels = train_df['label'].values
    test_labels = test_df['label'].values
    # 2.预处理
    prehandler = preprocessing.TextPreHandler(spliter=preprocessing.JiebaTextSpliter(),
                                              replacers=[preprocessing.Q2BTextReplacer()],
                                              removers=[preprocessing.ZhStopWordRemover()])
    # 3.模型训练
    tfidf_clf = TfidfClassifier(prehandler, min_df=5, ngram=1, C=1)
    tfidf_clf.train(train_text_list, train_labels)
    # 4.模型评估
    tfidf_clf.evaluation(test_text_list, test_labels)
