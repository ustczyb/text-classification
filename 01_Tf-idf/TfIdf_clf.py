from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

class TfidfClassifier(object):

    def __init__(self, min_df=5, token_level='char', ngram=1):
        self.token_level = token_level
        self.ngram = ngram
        self.vectorizer = TfidfVectorizer(analyzer=self.token_level, token_pattern="(?u)\b\w+\b", min_df=min_df)
        self.classifier = SVC(kernel='linear')
        self.has_trained = False


    def train(self, texts, labels):
        """
        模型训练
        :param texts:
        :param labels:
        :return:
        """
        # 1. 训练vectorizer
        self.vectorizer.fit(texts)
        train_X = self.vectorizer.transform(texts)
        self.classifier.fit(train_X, labels)
        self.has_trained = True

    def predict_single(self, text: str):
        text_vector = self.vectorizer.transform(text)
        return self.classifier.predict(text_vector)

