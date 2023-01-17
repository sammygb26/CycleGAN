import paramanager as pm
from cyclegan import models, util
import os

PROTO_PARAMETERS = [
    pm.ProtoParameter("main_folder", None, str, ["m"], True),
    pm.ProtoParameter("data_folder", None, str, ["d"], True),
    pm.ProtoParameter("load_folder", None, str, ["l"]),
] + util.TRAINING_PARAMS + util.ARCHITECTURE_PARAMS


def get_params() -> pm.ParameterSet:
    parameters = pm.ParameterSet(PROTO_PARAMETERS, "Parameters")
    parameters.read_argv()

    parameters.pretty_print()
    if input("Continue(y)?") != "y":
        quit()

    return parameters


def set_up_main_folder(main_folder: str):
    if not os.path.exists(main_folder):
        os.mkdir(main_folder)


if __name__ == "__main__":
    ps = get_params()
    cg = models.CycleGAN(ps, ps["load_folder"])
    mf = ps["main_folder"]

    set_up_main_folder(mf)

    data_a, data_b = util.get_datasets(*ps.get_all("data_folder", "batch_size"))
    trainer = util.CycleGANTrainer(cg, data_a, data_b, mf)

    trainer.epoch(0)


