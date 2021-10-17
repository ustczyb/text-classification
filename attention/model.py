import tensorflow.python.keras.models as models
import tensorflow.python.keras.layers as layers
import tensorflow.python.keras.regularizers as regularizers

class TextRNN(object):

    def __init__(self, vocab_size, embedding_size, hidden_size, max_sentence_len, dropout_prob=0.2):
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.max_sentence_len = max_sentence_len

        self.model = models.Sequential()
        # embedding层
        self.model.add(layers.Embedding(vocab_size, embedding_size, input_length=max_sentence_len, embeddings_regularizer=regularizers.l2(0.01)))
        # bi-LSTM层
        self.model.add(layers.Bidirectional(layers.LSTM(self.hidden_size, kernel_regularizer=regularizers.l2(0.01))))
        # dropout层
        self.model.add(layers.Dropout(dropout_prob))
        # 全连接
        self.model.add(layers.Dense(1, activation='sigmoid'))
        self.model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
        self.model.summary()

    def train(self, train_X, train_y, batch_size, epoch):
        self.model.fit(x=train_X, y=train_y, batch_size=batch_size, epochs=epoch, validation_split=0.2)

    def predict(self, x):
        return self.model.predict(x)

    def save_model(self, save_path):
        self.model.save(save_path)