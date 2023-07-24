from datetime import datetime
from lib.check_score import check_score_and_save, check_score_and_save_bin
from lib.fit import fit
from lib.load_model import load_model
from lib.model_compile import model_compile
from lib.set_dashboard import set_dashboard


def vit_fit(config, x_train, y_train, x_val, y_val, dashboard):
    model = load_model(config, x_train.shape[1:], len(set(y_train)))
    model = model_compile(model, config)

    print("----TRAINING SUMMARY----")
    print("SHAPE:\t", x_train.shape, "\tRANGE:\t", x_train.min(), x_train.max())
    print("CLASSES:\t", len(set(y_train)))
    start = datetime.now()
    model, history = fit(model, config, x_train, y_train, x_val, y_val, dashboard)
    end = datetime.now()
    return model, history
