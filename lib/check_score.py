import numpy as np
import pandas as pd
import sklearn.metrics as metrics
from datetime import datetime


def check_score_and_save(history, model, x_train, y_train, x_val, y_val, x_test, y_test, config, dashboard=None):
    scores = {"epoch_done": len(history.history["loss"])}

    res = np.argmax(model.predict(x_test), axis=-1)
    res_path = config["SETTINGS"]["OutputDir"]
    now = datetime.now()
    date = now.strftime("%Y/%m/%d %H:%M:%S")
    model.save(res_path + date + ".h5")

    scores["OA"] = metrics.accuracy_score(y_test, res) * 100
    scores["BalancedAccuracy"] = metrics.balanced_accuracy_score(y_test, res) * 100
    scores["F1W"] = metrics.f1_score(y_test, res, average="weighted") * 100
    scores["F1M"] = metrics.f1_score(y_test, res, average="macro") * 100
    scores = [history.history['val_loss'][epoch] for epoch in range(len(history.history['loss']))]
    scores["val_loss"] = min(scores)

    res_training = np.argmax(model.predict(np.array(np.concatenate((x_train, x_val), axis=0))), axis=-1)

    scores["OA_train"] = metrics.accuracy_score(np.concatenate((y_train, y_val), axis=0), res_training) * 100
    scores["BalancedAccuracy_train"] = metrics.balanced_accuracy_score(np.concatenate((y_train, y_val), axis=0),
                                                                       res_training) * 100
    scores["F1W_train"] = metrics.f1_score(np.concatenate((y_train, y_val), axis=0), res_training,
                                           average="weighted") * 100
    scores["F1M_train"] = metrics.f1_score(np.concatenate((y_train, y_val), axis=0), res_training,
                                           average="macro") * 100
    scores["cm"] = str(metrics.confusion_matrix(y_test, res))


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

    },

    df = pd.DataFrame(data=scores)
    df_conf = pd.DataFrame(data=config)
    #df.to_excel(, index=False)

    writer = pd.ExcelWriter(res_path + date +".xlsx", engine='xlsxwriter')
    df.to_excel(writer, sheet_name='results')
    df_conf.to_excel(writer, sheet_name='configuration')
    writer.save()
    writer.close()

    if dashboard is not None:
        dashboard.log(scores)
    return scores
