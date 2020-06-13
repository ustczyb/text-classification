from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

class TfidfClassifier(object):

    def __init__(self, min_df=5, token_level='char', ngram=1):
        self.token_level = token_level
        self.ngram = ngram
        self.vectorizer = TfidfVectorizer(analyzer=self.token_level, token_pattern="(?u)\b\w+\b", min_df=min_df)
        self.classifier = SVC(kernel='linear')


    def train(self, text, label):
        """
        模型训练
        :param text:
        :param label:
        :return:
        """
        # 1. 训练vectorizer
        self.vectorizer.fit(text)
        train_X = self.vectorizer.transform(text)
        self.classifier.fit(train_X, label)
