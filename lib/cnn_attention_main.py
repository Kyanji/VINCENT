import os
from datetime import datetime

import numpy as np
import pandas as pd
from hyperopt import Trials, hp, fmin, tpe, STATUS_OK
from keras.utils import to_categorical

from lib.check_score import check_score_and_save, check_score_and_save_bin
from lib.fit import fit
from lib.student_compile import student_compile
from lib.teacher_CNN import teacher_CNN
from lib.set_dashboard import set_dashboard, set_dashboard_distiller
import tensorflow as tf
from keras import backend as K

x_train = None
y_train = None
x_val = None
y_val = None
x_test = None
y_test = None

config = None
score_list = []
best_loss = None
now = datetime.now()
date = now.strftime("%Y-%m-%d-%H-%M-%S")
i = -1

best_model = None
best_score = None


def hyperopt_loop_cnn_attention(param):
    global x_train, y_train, x_val, y_val, x_test, y_test, i, config
    i = i + 1
    one_hot_encode_y_train = to_categorical(y_train, num_classes=len(set(y_train)))
    one_hot_encode_y_val = to_categorical(y_val, num_classes=len(set(y_test)))

    model = teacher_CNN(x_train.shape[1:], len(set(y_train)), param)
    model = student_compile(model, param)

    if config.getboolean("SETTINGS", "Wandb"):
        dashboard, wandb = set_dashboard_distiller(config, param, run_id=date, id=i)
    else:
        dashboard = []
        wandb = None
    callbacks = dashboard

    if config.getboolean("DISTILLATION", "EarlyStop"):
        stop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001,
                                                         patience=config.getint("DISTILLATION", "Patience"),
                                                         restore_best_weights=True, verbose=2)
        callbacks.append(stop_callback)
    start = datetime.now()

    history = model.fit(
        x=x_train,
        y=one_hot_encode_y_train,
        batch_size=param["batch"],
        epochs=config.getint("MODEL", "Epochs"),
        validation_data=(x_val, one_hot_encode_y_val),
        verbose=1,
        callbacks=callbacks,
    )

    print("end")
    # distiller.evaluate(x_with_h, y_test)
    if len(set(y_train)) != 2:
        scores, y_pred = check_score_and_save(history, model, x_train, y_train, x_val, y_val, x_test, y_test,
                                              config, save=False, distillation=False, dashboard=wandb,
                                              time=datetime.now() - start)
    else:
        scores = check_score_and_save_bin(history, model, x_train, y_train, x_val, y_val, x_test,
                                          y_test, config,
                                          len(set(y_train)), 0,
                                          save=False, distillation=False, dashboard=wandb, time=datetime.now() - start)
    scores.update(param)
    score_list.append(scores)
    if not os.path.exists(config[config["SETTINGS"]["Dataset"]]["OutputDir"] + "/cnn" + str(date)):
        os.makedirs(config[config["SETTINGS"]["Dataset"]]["OutputDir"] + "/cnn" + str(date))
    model.save_weights(
        config[config["SETTINGS"]["Dataset"]]["OutputDir"] + "/cnnAttention" + str(date) + "/" + str(i) + ".tf")
    global best_loss
    if best_loss is None:
        best_loss = score_list[-1]["val_loss"]
        model.save_weights(
            config[config["SETTINGS"]["Dataset"]]["OutputDir"] + "/cnnAttention" + str(date) + "/best.tf")
    elif score_list[-1]["val_loss"] < best_loss:
        best_loss = score_list[-1]["val_loss"]
        model.save_weights(
            config[config["SETTINGS"]["Dataset"]]["OutputDir"] + "/cnnAttention" + str(date) + "/best.tf")
    p = pd.DataFrame(score_list)
    p.to_excel(
        config[config["SETTINGS"]["Dataset"]]["OutputDir"] + "/cnnAttention" + str(date) + "/" + str(date) + ".xlsx")
    K.clear_session()
    if wandb is not None:
        wandb.finish()
    return {'loss': scores["val_loss"], 'status': STATUS_OK}


def cnn_attention_main(config2, x_train2, y_train2, x_val2, y_val2, x_test2, y_test2):
    global x_train, y_train, x_val, y_val, x_test, y_test, config
    x_train = x_train2
    y_train = y_train2
    x_val = x_val2
    y_val = y_val2
    x_test = x_test2
    y_test = y_test2
    config = config2

    print("----TRAINING SUMMARY----")
    print("SHAPE:\t", x_train.shape, "\tRANGE:\t", x_train.min(), x_train.max())
    print("CLASSES:\t", len(set(y_train)))

    trials = Trials()
    optimizable_variable = {
        "kernel": hp.choice("kernel", np.arange(2, 3 + 1)),
        "batch": hp.choice("batch", [64, 128, 256, 512]),
        'dropout1': hp.uniform("dropout1", 0, 1),
        'dropout2': hp.uniform("dropout2", 0, 1),
        "learning_rate": hp.uniform("learning_rate", 1e-4, 1e-1),
    }

    fmin(hyperopt_loop_cnn_attention, optimizable_variable, trials=trials, algo=tpe.suggest,
         max_evals=config.getint("DISTILLATION", "HyperoptEvaluations"))

    return best_model, best_score

    # model, history = fit(model, config, x_train, y_train, x_val, y_val, dashboard)
    # end = datetime.now()
    # return model, history
