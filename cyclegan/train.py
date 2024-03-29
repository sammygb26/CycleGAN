import paramanager as pm
import models
import util
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


def main():
    params = get_params()

    main_folder, epochs = params.get_all("main_folder", "epochs")
    set_up_main_folder(main_folder)
    params.write_file(os.path.join(main_folder, f"{params.name}.txt"))

    cycle_gan = models.CycleGAN(params, params["load_folder"])

    data_a, data_b = util.get_datasets(*params.get_all("data_folder", "train_ext_a", "train_ext_b", "batch_size"))
    trainer = util.CycleGANTrainer(cycle_gan, data_a, data_b, params)

    for i in range(epochs):
        trainer.epoch(i)

    print("done.")


if __name__ == "__main__":
    main()
