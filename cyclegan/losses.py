import paramanager as pm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


PROTO_PARAMETERS = [
    pm.ProtoParameter("losses_file", None, str, ["l"], True),
    pm.ProtoParameter("a", "loss_gen_a", str),
    pm.ProtoParameter("b", "loss_gen_b", str),
    pm.ProtoParameter("c", "loss_des_a", str),
    pm.ProtoParameter("d", "loss_des_b", str),
]


def get_params():
    params = pm.ParameterSet(PROTO_PARAMETERS, "Parameters")
    params.read_argv()

    params.pretty_print()
    if not input("Continue(y)?") == "y":
        quit()

    return params


def open_losses(file):
    return pd.read_csv(file)


def plot_losses(df: pd.DataFrame, params: pm.ParameterSet):
    x = np.arange(0, len(df), 1)

    plots = params.get_all("a", "b", "c", "d")
    for col in plots:
        if col:
            y = df[col]
            plt.plot(x, y, label=col)

    plt.legend()
    plt.show()




def main():
    params = get_params()
    df = open_losses(params["losses_file"])
    plot_losses(df, params)


if __name__ == "__main__":
    main()
