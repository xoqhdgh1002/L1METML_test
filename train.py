import tensorflow
import tensorflow.keras.backend as K
from tensorflow.keras import optimizers, initializers
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping, CSVLogger
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split

import numpy as np
import tables
import matplotlib.pyplot as plt
import argparse
import math
#import setGPU
import time
import os
import pathlib
import datetime
import tqdm
import h5py
from glob import glob
import itertools

# Import custom modules

from Write_MET_binned_histogram import *
from cyclical_learning_rate import CyclicLR
from models import *
from utils import *
from loss import custom_loss_wrapper
from DataGenerator import DataGenerator

import matplotlib.pyplot as plt
import mplhep as hep

def get_callbacks(path_out, sample_size, batch_size):
    # early stopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=40, verbose=1, restore_best_weights=False)

    csv_logger = CSVLogger(f'{path_out}loss_history.log')

    # model checkpoint callback
    # this saves our model architecture + parameters into model.h5
    model_checkpoint = ModelCheckpoint(f'{path_out}model.h5', monitor='val_loss',
                                       verbose=0, save_best_only=True,
                                       save_weights_only=False, mode='auto',
                                       period=1)

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=0.000001, cooldown=3, verbose=1)

    lr_scale = 1.
    clr = CyclicLR(base_lr=0.0003*lr_scale, max_lr=0.001*lr_scale, step_size=sample_size/batch_size, mode='triangular2')

    stop_on_nan = tensorflow.keras.callbacks.TerminateOnNaN()

    callbacks = [early_stopping, clr, stop_on_nan, csv_logger, model_checkpoint]

    return callbacks

def test(Yr_test, predict_test, PUPPI_pt, path_out):

    MakePlots(Yr_test, predict_test, PUPPI_pt, path_out=path_out)

    Yr_test = convertXY2PtPhi(Yr_test)
    predict_test = convertXY2PtPhi(predict_test)
    PUPPI_pt = convertXY2PtPhi(PUPPI_pt)

    extract_result(predict_test, Yr_test, path_out, 'TTbar', 'ML')
    extract_result(PUPPI_pt, Yr_test, path_out, 'TTbar', 'PU')

    MET_rel_error_opaque(predict_test[:, 0], PUPPI_pt[:, 0], Yr_test[:, 0], name=''+path_out+'rel_error_opaque.png')
    MET_binned_predict_mean_opaque(predict_test[:, 0], PUPPI_pt[:, 0], Yr_test[:, 0], 20, 0, 500, 0, '.', name=''+path_out+'PrVSGen.png')

    Phi_abs_error_opaque(PUPPI_pt[:, 1], predict_test[:, 1], Yr_test[:, 1], name=path_out+'Phi_abs_err')
    Pt_abs_error_opaque(PUPPI_pt[:, 0], predict_test[:, 0], Yr_test[:, 0], name=path_out+'Pt_abs_error')


def train_dataGenerator(args):
    # general setup
    maxNPF = args.maxNPF
    n_features_pf = 6
    n_features_pf_cat = 2
    normFac = 1.
    custom_loss = custom_loss_wrapper(normFac)
    epochs = args.epochs
    batch_size = args.batch_size
    preprocessed = True
    t_mode = args.mode
    inputPath = args.input
    path_out = args.output
    quantized = args.quantized
    model = args.model
    units = list(map(int, args.units))
    compute_ef = args.compute_edge_feat
    edge_list = args.edge_features

    # separate files into training, validation, and testing
    filesList = glob(os.path.join(inputPath, '*.root'))
    filesList.sort(reverse=True)

    assert len(filesList) >= 3, "Need at least 3 files for DataGenerator: 1 valid, 1 test, 1 train"

    valid_nfiles = max(1, int(.1*len(filesList)))
    train_nfiles = len(filesList) - 2*valid_nfiles
    test_nfiles = valid_nfiles
    train_filesList = filesList[0:train_nfiles]
    valid_filesList = filesList[train_nfiles: train_nfiles+valid_nfiles]
    test_filesList = filesList[train_nfiles+valid_nfiles:test_nfiles+train_nfiles+valid_nfiles]

    if compute_ef == 1:

        # set up data generators; they perform h5 conversion if necessary and load in data batch by batch
        trainGenerator = DataGenerator(list_files=train_filesList, batch_size=batch_size, maxNPF=maxNPF, compute_ef=1, edge_list=edge_list)
        validGenerator = DataGenerator(list_files=valid_filesList, batch_size=batch_size, maxNPF=maxNPF, compute_ef=1, edge_list=edge_list)
        testGenerator = DataGenerator(list_files=test_filesList, batch_size=batch_size, maxNPF=maxNPF, compute_ef=1, edge_list=edge_list)
        Xr_train, Yr_train = trainGenerator[0]  # this apparenly calls all the attributes, so that we can get the correct input dimensions (train_generator.emb_input_dim)

    else:
        trainGenerator = DataGenerator(list_files=train_filesList, batch_size=batch_size)
        validGenerator = DataGenerator(list_files=valid_filesList, batch_size=batch_size)
        testGenerator = DataGenerator(list_files=test_filesList, batch_size=batch_size)
        Xr_train, Yr_train = trainGenerator[0]  # this apparenly calls all the attributes, so that we can get the correct input dimensions (train_generator.emb_input_dim)

    # Load training model
    if quantized is None:
        if model == 'dense_embedding':
            keras_model = dense_embedding(n_features=n_features_pf,
                                          emb_out_dim=2,
                                          n_features_cat=n_features_pf_cat,
                                          activation='tanh',
                                          embedding_input_dim=trainGenerator.emb_input_dim,
                                          number_of_pupcandis=maxNPF,
                                          t_mode=t_mode,
                                          with_bias=False,
                                          units=units)
        elif model == 'graph_embedding':
            keras_model = graph_embedding(n_features=n_features_pf,
                                          emb_out_dim=2,
                                          n_features_cat=n_features_pf_cat,
                                          activation='tanh',
                                          embedding_input_dim=trainGenerator.emb_input_dim,
                                          number_of_pupcandis=maxNPF,
                                          units=units, compute_ef=compute_ef, edge_list=edge_list)

    else:
        logit_total_bits = int(quantized[0])
        logit_int_bits = int(quantized[1])
        activation_total_bits = int(quantized[0])
        activation_int_bits = int(quantized[1])

        keras_model = dense_embedding_quantized(n_features=n_features_pf,
                                                emb_out_dim=2,
                                                n_features_cat=n_features_pf_cat,
                                                activation_quantizer='quantized_relu',
                                                embedding_input_dim=trainGenerator.emb_input_dim,
                                                number_of_pupcandis=maxNPF,
                                                t_mode=t_mode,
                                                with_bias=False,
                                                logit_quantizer='quantized_bits',
                                                logit_total_bits=logit_total_bits,
                                                logit_int_bits=logit_int_bits,
                                                activation_total_bits=activation_total_bits,
                                                activation_int_bits=activation_int_bits,
                                                alpha=1,
                                                use_stochastic_rounding=False,
                                                units=units)

    # Check which model will be used (0 for L1MET Model, 1 for DeepMET Model)
    if t_mode == 0:
        keras_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error', 'mean_squared_error'])
        verbose = 1
    elif t_mode == 1:
        optimizer = optimizers.Adam(lr=1., clipnorm=1.)
        keras_model.compile(loss=custom_loss, optimizer=optimizer,
                            metrics=['mean_absolute_error', 'mean_squared_error'])
        verbose = 1

    # Run training
    print(keras_model.summary())

    start_time = time.time()  # check start time
    history = keras_model.fit(trainGenerator,
                              epochs=epochs,
                              verbose=verbose,  # switch to 1 for more verbosity
                              validation_data=validGenerator,
                              callbacks=get_callbacks(path_out, len(trainGenerator), batch_size))
    end_time = time.time()  # check end time

    predict_test = keras_model.predict(testGenerator) * normFac
    all_PUPPI_pt = []
    Yr_test = []
    for (Xr, Yr) in tqdm.tqdm(testGenerator):
        puppi_pt = np.sum(Xr[1], axis=1)
        all_PUPPI_pt.append(puppi_pt)
        Yr_test.append(Yr)

    PUPPI_pt = normFac * np.concatenate(all_PUPPI_pt)
    Yr_test = normFac * np.concatenate(Yr_test)

    test(Yr_test, predict_test, PUPPI_pt, path_out)

    fi = open("{}time.txt".format(path_out), 'w')

    fi.write("Working Time (s) : {}".format(end_time - start_time))
    fi.write("Working Time (m) : {}".format((end_time - start_time)/60.))

    fi.close()

def main():
    time_path = time.strftime('%Y-%m-%d', time.localtime(time.time()))

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--workflowType',
        action='store',
        type=str,
        required=True,
        choices=[
            'dataGenerator',
            'loadAllData'],
        help='designate wheather youre using the data generator or loading all data into memory ')
    parser.add_argument('--input', action='store', type=str, required=True, help='designate input file path')
    parser.add_argument('--output', action='store', type=str, required=True, help='designate output file path')
    parser.add_argument('--mode', action='store', type=int, required=True, choices=[0, 1], help='0 for L1MET, 1 for DeepMET')
    parser.add_argument('--epochs', action='store', type=int, required=False, default=100, help='number of epochs to train for')
    parser.add_argument('--batch-size', action='store', type=int, required=False, default=1024, help='batch size')
    parser.add_argument('--quantized', action='store', required=False, nargs='+', help='optional argument: flag for quantized model and specify [total bits] [int bits]; empty for normal model')
    parser.add_argument('--units', action='store', required=False, nargs='+', help='optional argument: specify number of units in each layer (also sets the number of layers)')
    parser.add_argument('--model', action='store', required=False, choices=['dense_embedding', 'graph_embedding', 'node_select'], default='dense_embedding', help='optional argument: model')
    parser.add_argument('--compute-edge-feat', action='store', type=int, required=False, choices=[0, 1], default=0, help='0 for no edge features, 1 to include edge features')
    parser.add_argument('--maxNPF', action='store', type=int, required=False, default=100, help='maximum number of PUPPI candidates')
    parser.add_argument('--edge-features', action='store', required=False, nargs='+', help='which edge features to use (i.e. dR, kT, z, m2)')

    args = parser.parse_args()
    workflowType = args.workflowType

    os.makedirs(args.output, exist_ok=True)

    if workflowType == 'dataGenerator':
        train_dataGenerator(args)
    elif workflowType == 'loadAllData':
        train_loadAllData(args)


if __name__ == "__main__":
    main()
