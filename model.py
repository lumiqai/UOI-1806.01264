import keras
from keras_global_self_attention import Attention
from keras_contrib.layers import CRF


def build_model(token_num,
                tag_num,
                embedding_dim=100,
                embedding_weights=None,
                rnn_units=100):
    """Build the model for predicting tags.

    :param token_num: Number of tokens in the word dictionary.
    :param tag_num: Number of tags.
    :param embedding_dim: The output dimension of the embedding layer.
    :param embedding_weights: Initial weights for embedding layer.
    :param rnn_units: The number of RNN units in a single direction.
    :return model: The built model.
    """
    if embedding_weights is not None and not isinstance(embedding_weights, list):
        embedding_weights = [embedding_weights]

    model = keras.models.Sequential()
    model.add(keras.layers.Embedding(input_dim=token_num,
                                     output_dim=embedding_dim,
                                     mask_zero=True,
                                     weights=embedding_weights,
                                     trainable=embedding_weights is None,
                                     name='Embedding'))
    model.add(keras.layers.Bidirectional(keras.layers.LSTM(units=rnn_units,
                                                           recurrent_dropout=0.4,
                                                           return_sequences=True),
                                         name='Bi-LSTM'))
    model.add(Attention(name='Attention'))
    crf = CRF(units=tag_num, sparse_target=True, name='CRF')
    model.add(crf)

    model.compile(
        optimizer='adam',
        loss=crf.loss_function,
        metrics=[crf.accuracy],
    )
    return model
