import numpy as np
import pandas as pd
import sklearn.metrics as metrics
from datetime import datetime

from keras.utils import to_categorical


# from torch.testing._internal.distributed.rpc.examples.parameter_server_test import batch_size


def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


def check_score_and_save(history, model, x_train, y_train, x_val, y_val, x_test, y_test, config, dashboard=None,
                         time=None, save=True, distillation=False):
    if distillation:
        scores = {"epoch_done": len(history.history["val_student_loss"])}

    else:
        scores = {"epoch_done": len(history.history["loss"])}

    # res = np.argmax(model.predict(x_test), axis=-1)
    index = list(split(range(len(x_test)), 10))
    res_partial = [np.argmax(model.predict(np.array(x_test[k])), axis=-1) for k in index]
    res = []
    for r in res_partial:
        res.extend(r)
    res_path = config[config["SETTINGS"]["Dataset"]]["OutputDir"]
    now = datetime.now()
    date = now.strftime("%Y-%m-%d-%H-%M-%S")
    if save:
        model.save(res_path + date + ".h5")

    scores["OA"] = metrics.accuracy_score(y_test, res) * 100
    scores["BalancedAccuracy"] = metrics.balanced_accuracy_score(y_test, res) * 100
    scores["F1W"] = metrics.f1_score(y_test, res, average="weighted") * 100
    scores["F1M"] = metrics.f1_score(y_test, res, average="macro") * 100
    if distillation:
        distillation_loss = [history.history['distillation_loss'][epoch] for epoch in
                             range(len(history.history['distillation_loss']))]
        scores["distillation_loss"] = min(distillation_loss)
        val_student_loss = [history.history['val_student_loss'][epoch] for epoch in
                            range(len(history.history['val_student_loss']))]
        scores["val_student_loss"] = min(val_student_loss)
    else:
        scores_loss = [history.history['val_loss'][epoch] for epoch in range(len(history.history['loss']))]
        scores["val_loss"] = min(scores_loss)
    index = list(split(range(len(x_train)), 8))
    res_training = []
    for k in index:
        try:
            total_x = np.argmax(model.predict(np.array(x_train[k])), axis=-1)
            res_training.extend(total_x)
        except Exception as e:
            print("Out of Memory"+str(e))
            pass
    index = list(split(range(len(x_val)), 5))
    total_x = [np.argmax(model.predict(np.array(x_val[k])), axis=-1) for k in index]
    total_x_full = []
    for r in total_x:
        total_x_full.extend(r)
    res_training.extend(total_x_full)

    scores["OA_train"] = metrics.accuracy_score(np.concatenate((y_train, y_val), axis=0), res_training) * 100
    scores["BalancedAccuracy_train"] = metrics.balanced_accuracy_score(np.concatenate((y_train, y_val), axis=0),
                                                                       res_training) * 100
    scores["F1W_train"] = metrics.f1_score(np.concatenate((y_train, y_val), axis=0), res_training,
                                           average="weighted") * 100
    scores["F1M_train"] = metrics.f1_score(np.concatenate((y_train, y_val), axis=0), res_training,
                                           average="macro") * 100
    scores["cm"] = str(metrics.confusion_matrix(y_test, res))
    if time is not None:
        scores["time(m)"] = time.total_seconds() / 60

    config = {
        # "learning_rate": 0.02,
        "UseRGBEncoding": config["SETTINGS"]["UseRGBEncoding"],
        "Dataset": config["SETTINGS"]["Dataset"],
        "Lr": config["MODEL"]["Lr"],
        "Decay": config["MODEL"]["Decay"],
        "EarlyStop": config["MODEL"]["EarlyStop"],
        "Patience": config["MODEL"]["Patience"],
        "BatchSize": config["MODEL"]["BatchSize"],
        "Epochs": config["MODEL"]["Epochs"],

        "HiddenDim": config["VIT_SETTINGS"]["HiddenDim"],
        "PatchSize": config["VIT_SETTINGS"]["PatchSize"],
        "NumLayer": config["VIT_SETTINGS"]["NumLayer"],
        "NumHeads": config["VIT_SETTINGS"]["NumHeads"],
        "MlpDim": config["VIT_SETTINGS"]["MlpDim"],
        "Dropout": config["VIT_SETTINGS"]["Dropout"],
        "Seed": config["SETTINGS"]["Seed"],
    }
    df = pd.DataFrame(data=scores, index=[1])
    df_conf = pd.DataFrame(data=config, index=[1])
    # df.to_excel(, index=False)
    if save:
        writer = pd.ExcelWriter(res_path + date + ".xlsx", engine='xlsxwriter')
        df.to_excel(writer, sheet_name='results')
        df_conf.to_excel(writer, sheet_name='configuration')
        writer.save()
        writer.close()

    if dashboard is not None:
        dashboard.log(scores)
    return scores, res


def check_score_and_save_bin(history, model, x_train, y_train, x_val, y_val, x_test, y_test, config, N_CLASSES, time,
                             dashboard=None, save=True, distillation=False):
    res = np.argmax(model.predict(x_test), axis=-1)
    res_path = config[config["SETTINGS"]["Dataset"]]["OutputDir"]
    now = datetime.now()
    date = now.strftime("%Y-%m-%d-%H-%M-%S")
    if save:
        model.save(res_path + date + ".h5")

    cm = metrics.confusion_matrix(y_test, res)
    tp = cm[0][0]  # attacks true
    fn = cm[0][1]  # attacs predict normal
    fp = cm[1][0]  # normal predict attacks
    tn = cm[1][1]  # normal as normal
    attacks = tp + fn
    normals = fp + tn
    OA = (tp + tn) / (attacks + normals)
    AA = ((tp / attacks) + (tn / normals)) / N_CLASSES
    P = tp / (tp + fp)
    R = tp / (tp + fn)
    F1 = 2 * ((P * R) / (P + R))
    FAR = fp / (fp + tn)
    TPR = tp / (tp + fn)
    r = [tp, fn, fp, tn, OA, AA, P, R, F1, FAR, TPR]
    scores = {}
    scores["tp"] = tp
    scores["fn"] = fn
    scores["fp"] = fp
    scores["tn"] = tn
    scores["attacks"] = attacks
    scores["normals"] = normals

    scores["OA"] = OA
    scores["AA"] = AA
    scores["P"] = P
    scores["R"] = R
    scores["F1"] = F1
    scores["FAR"] = FAR
    scores["TPR"] = TPR

    scores_loss = [history.history['val_loss'][epoch] for epoch in range(len(history.history['loss']))]
    scores["val_loss"] = min(scores_loss)
    scores["val_loss_last"] = history.history['val_loss'][-1]
    scores["val_loss_best_epoch"] = np.argmin(history.history['val_loss'])
    if distillation:
        distillation_loss = [history.history['distillation_loss'][epoch] for epoch in
                             range(len(history.history['distillation_loss']))]
        scores["distillation_loss"] = min(distillation_loss)

    scores["epoch"] = len(history.history["val_loss"])

    res_training = np.argmax(model.predict(np.array(np.concatenate((x_train, x_val), axis=0))), axis=-1)
    cm = metrics.confusion_matrix(np.array(np.concatenate((y_train, y_val))), res_training)
    tp = cm[0][0]  # attacks true
    fn = cm[0][1]  # attacs predict normal
    fp = cm[1][0]  # normal predict attacks
    tn = cm[1][1]  # normal as normal
    attacks = tp + fn
    normals = fp + tn
    OA = (tp + tn) / (attacks + normals)
    AA = ((tp / attacks) + (tn / normals)) / N_CLASSES
    P = tp / (tp + fp)
    R = tp / (tp + fn)
    F1 = 2 * ((P * R) / (P + R))
    FAR = fp / (fp + tn)
    TPR = tp / (tp + fn)
    rtrain = [tp, fn, fp, tn, OA, AA, P, R, F1, FAR, TPR]

    scores_t = {}
    scores_t["tp_t"] = tp
    scores_t["fn_t"] = fn
    scores_t["fp_t"] = fp
    scores_t["tn_t"] = tn
    scores_t["attacks_t"] = attacks
    scores_t["normals_t"] = normals

    scores_t["OA_t"] = OA
    scores_t["AA_t"] = AA
    scores_t["P_t"] = P
    scores_t["R_t"] = R
    scores_t["F1_t"] = F1
    scores_t["FAR_t"] = FAR
    scores_t["TPR_t"] = TPR

    scores["time(m)"] = time.total_seconds() / 60

    df = pd.DataFrame(data=scores, index=[1])

    config = {
        # "learning_rate": 0.02,
        "UseRGBEncoding": config["SETTINGS"]["UseRGBEncoding"],
        "Dataset": config["SETTINGS"]["Dataset"],
        "UseScale0_1": config["SETTINGS"]["UseScale0_1"],
        "UseScale-1_1": config["SETTINGS"]["UseScale-1_1"],
        "Resize": config["SETTINGS"]["Resize"],
        "ResizeShape": config["SETTINGS"]["ResizeShape"],
        "Model": config["MODEL"]["Model"],
        "Lr": config["MODEL"]["Lr"],
        "Decay": config["MODEL"]["Decay"],
        "EarlyStop": config["MODEL"]["EarlyStop"],
        "Patience": config["MODEL"]["Patience"],
        "BatchSize": config["MODEL"]["BatchSize"],
        "Epochs": config["MODEL"]["Epochs"],

        "HiddenDim": config["VIT_SETTINGS"]["HiddenDim"],
        "PatchSize": config["VIT_SETTINGS"]["PatchSize"],
        "NumLayer": config["VIT_SETTINGS"]["NumLayer"],
        "NumHeads": config["VIT_SETTINGS"]["NumHeads"],
        "MlpDim": config["VIT_SETTINGS"]["MlpDim"],
        "Dropout": config["VIT_SETTINGS"]["Dropout"],
        "Seed": config["SETTINGS"]["Seed"],
    }
    df_scores_t = pd.DataFrame(data=scores_t, index=[1])
    df_conf = pd.DataFrame(data=config, index=[1])
    if save:
        writer = pd.ExcelWriter(res_path + date + ".xlsx", engine='xlsxwriter')
        df.to_excel(writer, sheet_name='results')
        df_scores_t.to_excel(writer, sheet_name='results_train')
        df_conf.to_excel(writer, sheet_name='configuration')
        writer.save()
        writer.close()

    if dashboard is not None:
        dashboard.log(scores)
    return scores
