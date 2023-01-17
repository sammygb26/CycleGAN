import paramanager as pm
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import torch
import models
import util

PROTO_PARAMETERS = [
    pm.ProtoParameter("load_folder", None, str, ["l"], True),
    pm.ProtoParameter("data_folder", None, str, ["d"], True),
    pm.ProtoParameter("n_show", 5, int, ["n"])
] + util.ARCHITECTURE_PARAMS


def get_params() -> pm.ParameterSet:
    params = pm.ParameterSet(PROTO_PARAMETERS, "Parameters")
    params.read_argv()

    params.pretty_print()
    if not input("Continue(y)?") == "y":
        quit()

    return params


def get_cycle_gan(params: pm.ParameterSet):
    print("Loading model...")
    return models.CycleGAN(params, load_folder=params["load_folder"], training=False)


def get_images(params: pm.ParameterSet):
    data_a, data_b = util.get_datasets(*params.get_all("data_folder", "n_show"))
    return next(iter(data_a))[0], next(iter(data_b))[0]


def generate_images(faker, un_faker, real):
    fake = faker(real)
    rec = un_faker(fake)

    return fake, rec


def generate_all_images(cycle_gan: models.CycleGAN, real_a, real_b):
    print("Generating images...")
    fake_a, rec_b = generate_images(cycle_gan.gen_a, cycle_gan.gen_b, real_b)
    fake_b, rec_a = generate_images(cycle_gan.gen_b, cycle_gan.gen_a, real_a)

    return fake_a, fake_b, rec_a, rec_b


def display_images(real_a, real_b, fake_a, fake_b, rec_a, rec_b, n_show):
    print("Displaying images...")
    images = torch.cat((real_a, fake_b, rec_a, real_b, fake_a, rec_b)).cpu()
    images = images * 0.5 + 0.5
    plt.imshow(torch.permute(make_grid(images, n_show), (1, 2, 0)))
    plt.show()


def main():
    params = get_params()
    cycle_gan = get_cycle_gan(params)

    real_a, real_b = get_images(params)
    fake_a, fake_b, rec_a, rec_b = generate_all_images(cycle_gan, real_a, real_b)

    display_images(real_a, real_b, fake_a, fake_b, rec_a, rec_b, params["n_show"])



if __name__ == "__main__":
    main()


