from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, BatchNormalization, Dropout, Lambda, Conv1D, SpatialDropout1D, Concatenate, Flatten, Reshape, Multiply, Add, GlobalAveragePooling1D, Activation, Permute
import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow import slice
from tensorflow.keras import initializers
import qkeras
from qkeras.qlayers import QDense, QActivation
import numpy as np
import itertools


def dense_embedding(n_features=6,
                    n_features_cat=2,
                    activation='relu',
                    number_of_pupcandis=100,
                    embedding_input_dim={0: 13, 1: 3},
                    emb_out_dim=8,
                    with_bias=True,
                    t_mode=0,
                    units=[64, 32, 16]):
    n_dense_layers = len(units)

    inputs_cont = Input(shape=(number_of_pupcandis, n_features-2), name='input_cont')
    pxpy = Input(shape=(number_of_pupcandis, 2), name='input_pxpy')

    embeddings = []
    inputs = [inputs_cont, pxpy]
    for i_emb in range(n_features_cat):
        input_cat = Input(shape=(number_of_pupcandis, ), name='input_cat{}'.format(i_emb))
        inputs.append(input_cat)
        embedding = Embedding(
            input_dim=embedding_input_dim[i_emb],
            output_dim=emb_out_dim,
            embeddings_initializer=initializers.RandomNormal(
                mean=0,
                stddev=0.4/emb_out_dim),
            name='embedding{}'.format(i_emb))(input_cat)
        embeddings.append(embedding)

    emb_concat = Concatenate()(embeddings)
    x = Concatenate()([inputs_cont, emb_concat])

    for i_dense in range(n_dense_layers):
        x = Dense(units[i_dense], activation='linear', kernel_initializer='lecun_uniform')(x)
        x = BatchNormalization(momentum=0.95)(x)
        x = Activation(activation=activation)(x)

    if t_mode == 0:
        x = GlobalAveragePooling1D(name='pool')(x)
        x = Dense(2, name='output', activation='linear')(x)

    if t_mode == 1:
        if with_bias:
            b = Dense(2, name='met_bias', activation='linear', kernel_initializer=initializers.VarianceScaling(scale=0.02))(x)
            pxpy = Add()([pxpy, b])
        w = Dense(1, name='met_weight', activation='linear', kernel_initializer=initializers.VarianceScaling(scale=0.02))(x)
        w = BatchNormalization(trainable=False, name='met_weight_minus_one', epsilon=False)(w)
        x = Multiply()([w, pxpy])

        x = GlobalAveragePooling1D(name='output')(x)
    outputs = x

    keras_model = Model(inputs=inputs, outputs=outputs)

    keras_model.get_layer('met_weight_minus_one').set_weights([np.array([1.]), np.array([-1.]), np.array([0.]), np.array([1.])])

    return keras_model

def test_model(n_features=6,
               n_features_cat=2,
               activation='relu',
               number_of_pupcandis=100,
               embedding_input_dim={0: 13, 1: 3},
               emb_out_dim=8,
               with_bias=True,
               t_mode=0,
               units=[64, 32, 16]):

    n_dense_layers = len(units)

    inputs_cont = Input(shape=(number_of_pupcandis, n_features-2), name='input_cont')
    pxpy = Input(shape=(number_of_pupcandis, 2), name='input_pxpy')

    embeddings = []
    inputs = [inputs_cont, pxpy]
    for i_emb in range(n_features_cat):
        input_cat = Input(shape=(number_of_pupcandis, ), name='input_cat{}'.format(i_emb))
        inputs.append(input_cat)
        embedding = Embedding(
            input_dim=embedding_input_dim[i_emb],
            output_dim=emb_out_dim,
            embeddings_initializer=initializers.RandomNormal(
                mean=0,
                stddev=0.4/emb_out_dim),
            name='embedding{}'.format(i_emb))(input_cat)
        embeddings.append(embedding)

    emb_concat = Concatenate()(embeddings)
    x = Concatenate()([inputs_cont, emb_concat])
