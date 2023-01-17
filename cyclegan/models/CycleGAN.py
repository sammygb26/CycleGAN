import cyclegan.models as models
import itertools
import torch
from torch import nn
import paramanager as pm
import os


device = "cuda" if torch.cuda.is_available() else "cpu"


def backward_des(des_real, des_fake):
    real_loss = torch.mean((1 - des_real) ** 2)
    fake_loss = torch.mean(des_fake ** 2)

    return real_loss + fake_loss


def backward_gen(real, fake, rec, des_score, lambda_cyc, lambda_idt):
    cyc_loss = torch.mean(torch.abs(rec - real)) * lambda_cyc
    idt_loss = torch.mean(torch.abs(fake - real)) * lambda_idt
    gan_loss = torch.mean((1 - des_score) ** 2)

    return cyc_loss + idt_loss + gan_loss


def get_model_folders(load_folder: str = None):
    if not load_folder:
        return None, None, None, None

    return [os.path.join(load_folder, e + ".pth") for e in ["gen_a", "gen_b", "des_a", "des_b"]]


def set_requires_grad(nets: list[nn.Module], requires_grad: bool):
    for net in nets:
        net.requires_grad_(requires_grad)


class CycleGAN:
    def __init__(self, params: pm.ParameterSet, load_folder: str = None, training: bool = True):
        self.params = params

        # Setting up class values
        self.loss_des_b, self.loss_des_a, self.loss_gen_a, self.loss_gen_b, self.loss = [0.0] * 5
        self.des_fake_a, self.des_fake_b, self.des_real_a, self.des_real_b = [None] * 4
        self.real_a, self.real_b, self.fake_a, self.fake_b, self.rec_a, self.rec_b = [None] * 6

        # Get params
        nc_a, nc_b, ngf, n_down, n_res, ndf = params.get_all("nc_a", "nc_b", "ngf", "n_down", "n_res", "ndf")
        self.lambda_cyc, self.lambda_idt = self.params.get_all("lambda_cyc", "lambda_idt")

        gen_a_path, gen_b_path, des_a_path, des_b_path = get_model_folders(load_folder)

        # Generators
        self.gen_a = models.Generator(nc_b, nc_a, ngf, n_down, n_res, gen_a_path).to(device)
        self.gen_b = models.Generator(nc_a, nc_b, ngf, n_down, n_res, gen_b_path).to(device)

        if training:
            # Discriminators (only needed when training)
            self.des_a = models.Discriminator(nc_a, ndf, des_a_path).to(device)
            self.des_b = models.Discriminator(nc_b, ndf, des_b_path).to(device)

            lr, beta1, beta2 = params.get_all("lr", "beta1", "beta2")

            # Set up optimizers
            self.optimizer_gen = torch.optim.Adam(
                itertools.chain(self.gen_a.parameters(), self.gen_b.parameters()),
                lr, (beta1, beta2))
            self.optimizer_des = torch.optim.Adam(itertools.chain(
                self.des_a.parameters(), self.des_b.parameters()),
                lr, (beta1, beta2))

    def set_input(self, img_a: torch.Tensor, img_b: torch.Tensor):
        img_a.to(device)
        img_b.to(device)

        self.real_a = img_a
        self.real_b = img_b

    def forward_gen(self):
        self.fake_a = self.gen_a(self.real_b)
        self.fake_b = self.gen_b(self.real_a)
        self.rec_a = self.gen_a(self.fake_b)
        self.rec_b = self.gen_b(self.fake_a)

    def forward_des(self):
        self.des_fake_a = self.des_a(self.fake_a)
        self.des_fake_b = self.des_b(self.fake_b)

        self.des_real_a = self.des_a(self.real_a)
        self.des_real_b = self.des_b(self.real_b)

    def backward(self):
        set_requires_grad([self.des_a, self.des_b], False)

        # gen_a & gen_b
        self.loss_gen_a = backward_gen(
            self.real_a, self.fake_a, self.rec_a, self.des_fake_a, self.lambda_cyc, self.lambda_idt)
        self.loss_gen_b = backward_gen(
            self.real_b, self.fake_b, self.rec_b, self.des_fake_b, self.lambda_cyc, self.lambda_idt)

        set_requires_grad([self.des_a, self.des_b], True)
        set_requires_grad([self.gen_a, self.gen_b], False)

        # des_a & des_b
        self.loss_des_a = backward_des(self.des_real_a, self.des_fake_a)
        self.loss_des_b = backward_des(self.des_real_b, self.des_fake_b)

        set_requires_grad([self.gen_a, self.gen_b], True)

        self.loss = self.loss_gen_a + self.loss_gen_b + self.loss_des_a + self.loss_des_b
        self.loss.backward()

    def optimize_parameters(self):
        self.forward_gen()
        self.forward_des()

        self.optimizer_gen.zero_grad()
        self.optimizer_des.zero_grad()

        self.backward()

        self.optimizer_gen.step()
        self.optimizer_des.step()

    def save(self, folder: str):
        if not os.path.exists(folder):
            os.mkdir(folder)

        torch.save(self.gen_a.state_dict(), f"{folder}/gen_a.pth")
        torch.save(self.gen_b.state_dict(), f"{folder}/gen_b.pth")
        torch.save(self.des_a.state_dict(), f"{folder}/des_a.pth")
        torch.save(self.des_b.state_dict(), f"{folder}/des_b.pth")
