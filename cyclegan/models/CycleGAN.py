import itertools
import torch
from torch import nn
import paramanager as pm
from .Generator import Generator
from .Discriminator import Discriminator
from .ImagePool import ImagePool
import os


device = "cuda" if torch.cuda.is_available() else "cpu"


def backward_des(des, real, fake):
    real_loss = torch.mean((1 - des(real)) ** 2)
    fake_loss = torch.mean(des(fake) ** 2)

    return real_loss + fake_loss


def get_model_folders(load_folder:(None|str) = None):
    if not load_folder:
        return None, None, None, None

    return [os.path.join(load_folder, e + ".pth") for e in ["gen_a", "gen_b", "des_a", "des_b"]]


def set_requires_grad(nets: list[nn.Module], requires_grad: bool):
    for net in nets:
        if net:
            for param in net.parameters():
                param.requires_grad = requires_grad


class CycleGAN:
    def __init__(self, params: pm.ParameterSet, load_folder: (None|str) = None, training: bool = True):
        self.params = params

        # Setting up class values
        self.loss_des_b, self.loss_des_a, self.loss_gen_a, self.loss_gen_b, self.loss = [0.0] * 5
        self.real_a, self.real_b, self.fake_a, self.fake_b, self.rec_a, self.rec_b = [torch.Tensor([0])] * 6

        # Get params
        nc_a, nc_b, ngf, n_down, n_res, ndf = params.get_all("nc_a", "nc_b", "ngf", "n_down", "n_res", "ndf")
        self.lambda_cyc, self.lambda_idt = self.params.get_all("lambda_cyc", "lambda_idt")

        gen_a_path, gen_b_path, des_a_path, des_b_path = get_model_folders(load_folder)

        # Generators
        self.gen_a = Generator(nc_b, nc_a, ngf, n_down, n_res, gen_a_path).to(device)
        self.gen_b = Generator(nc_a, nc_b, ngf, n_down, n_res, gen_b_path).to(device)

        if training:
            # Discriminators (only needed when training)
            self.des_a = Discriminator(nc_a, ndf, des_a_path).to(device)
            self.des_b = Discriminator(nc_b, ndf, des_b_path).to(device)

            lr, beta1, beta2, pool_size = params.get_all("lr", "beta1", "beta2", "pool_size")

            self.image_pool = ImagePool(pool_size)

            # Set up optimizers
            self.optimizer_gen = torch.optim.Adam(
                itertools.chain(self.gen_a.parameters(), self.gen_b.parameters()),
                lr, (beta1, beta2))
            self.optimizer_des = torch.optim.Adam(itertools.chain(
                self.des_a.parameters(), self.des_b.parameters()),
                lr, (beta1, beta2))

    def set_input(self, img_a: torch.Tensor, img_b: torch.Tensor):
        self.real_a = img_a.to(device)
        self.real_b = img_b.to(device)

    def forward_gen(self):
        self.fake_a : torch.Tensor = self.gen_a(self.real_b)
        self.fake_b : torch.Tensor = self.gen_b(self.real_a)

        self.rec_a = self.gen_a(self.fake_b)
        self.rec_b = self.gen_b(self.fake_a)

    def backward(self):
        set_requires_grad([self.des_a, self.des_b], False)
        self.backward_gen()
        set_requires_grad([self.des_a, self.des_b], True)

        self.backward_des()
        self.loss = self.loss_gen_a + self.loss_gen_b + self.loss_des_a + self.loss_des_b

    def backward_des(self):
        fake_a = self.image_pool.query(self.fake_a.detach())
        fake_b = self.image_pool.query(self.fake_b.detach())

        # des_a & des_b
        self.loss_des_a = backward_des(self.des_a, self.real_a, fake_a)
        self.loss_des_b = backward_des(self.des_b, self.real_b, fake_b)

        # loss is halved to reduce rate at which D learns relative to G
        self.optimizer_des.zero_grad()
        (self.loss_des_a + self.loss_des_b).backward()
        self.optimizer_des.step()

    def backward_gen(self):
        # gen_a & gen_b
        if self.lambda_idt > 0:
            loss_idt_a = torch.mean(torch.abs(self.real_b - self.fake_a)) * self.lambda_idt
            loss_idt_b = torch.mean(torch.abs(self.real_a - self.fake_b)) * self.lambda_idt
        else:
            loss_idt_a = 0
            loss_idt_b = 0

        loss_gan_a = torch.mean((1 - self.des_a(self.fake_a)) ** 2)
        loss_gan_b = torch.mean((1 - self.des_b(self.fake_b)) ** 2)

        loss_cyc_a = torch.mean(torch.abs(self.real_a - self.rec_a)) * self.lambda_cyc
        loss_cyc_b = torch.mean(torch.abs(self.real_b - self.rec_b)) * self.lambda_cyc

        self.loss_gen_a = loss_gan_a + loss_cyc_a + loss_idt_a
        self.loss_gen_b = loss_gan_b + loss_cyc_b + loss_idt_b

        self.optimizer_gen.zero_grad()
        (self.loss_gen_a + self.loss_gen_b).backward()
        self.optimizer_gen.step()

    def optimize_parameters(self):
        self.forward_gen()
        self.backward()

    def save(self, folder: str):
        if not os.path.exists(folder):
            os.mkdir(folder)

        torch.save(self.gen_a.state_dict(), f"{folder}/gen_a.pth")
        torch.save(self.gen_b.state_dict(), f"{folder}/gen_b.pth")
        torch.save(self.des_a.state_dict(), f"{folder}/des_a.pth")
        torch.save(self.des_b.state_dict(), f"{folder}/des_b.pth")
