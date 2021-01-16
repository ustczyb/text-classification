import tensorflow as tf
import tensorflow.python.keras.models as models
import tensorflow.python.keras.layers as layers

class Fasttext(object):

    def __init__(self, vocab_size, embedding_size, max_sentence_len):
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        # create model
        self.model = models.Sequential()
        self.model.add(layers.Embedding(vocab_size, embedding_size, max_sentence_len))
        self.model.add(layers.GlobalAveragePooling1D())
        self.model.add(layers.Dense(units=1, activation='sigmoid'))
        # 损失和优化器选择
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model.summary()


    def train(self, train_X, train_y, batch_size, epoch):
        self.model.fit(train_X, train_y, )