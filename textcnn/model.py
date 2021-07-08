import tensorflow.python.keras.models as models
import tensorflow.python.keras.layers as layers

class TextCNN(object):

    def __init__(self, vocab_size, embedding_size, max_sentence_len, filter_nums):
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        # create model
        # input层
        input_layer = layers.Input(shape=(max_sentence_len,), dtype='int32')
        # embedding层
        embedding_layer = layers.Embedding(vocab_size, embedding_size, input_length=max_sentence_len)(input_layer)
        # 卷积-池化层
        conv1_layer = layers.convolutional.Conv1D(filters=filter_nums, kernel_size=3, padding='valid', activation='relu')(embedding_layer)
        pool1_layer = layers.MaxPooling1D(max_sentence_len - 3 + 1)(conv1_layer)

        conv2_layer = layers.convolutional.Conv1D(filters=filter_nums, kernel_size=4, padding='valid', activation='relu')(embedding_layer)
        pool2_layer = layers.MaxPooling1D(max_sentence_len - 4 + 1)(conv2_layer)

        conv3_layer = layers.convolutional.Conv1D(filters=filter_nums, kernel_size=5, padding='valid', activation='relu')(embedding_layer)
        pool3_layer = layers.MaxPooling1D(max_sentence_len - 5 + 1)(conv3_layer)
        # 池化向量拼接-全连接层
        concat_pool_layer = layers.Flatten()(layers.merge.concatenate([pool1_layer, pool2_layer, pool3_layer], axis=-1))
        dropout_layer = layers.Dropout(0.2)(concat_pool_layer)
        output_layer = layers.Dense(1, activation='sigmoid')(dropout_layer)

        self.model = models.Model(inputs=input_layer, outputs=output_layer, name='TextCNN')
        # 损失和优化器选择
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model.summary()


    def train(self, train_X, train_y, batch_size, epoch):
        self.model.fit(x=train_X, y=train_y, batch_size=batch_size, epochs=epoch, validation_split=0.2)

    def predict(self, x):
        return self.model.get_layer().predict(x)

    def save_model(self, save_path):
        self.model.save(save_path)