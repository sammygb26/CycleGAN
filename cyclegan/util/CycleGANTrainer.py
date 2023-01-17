from cyclegan import models
import torch
import os


device = "cuda" if torch.cuda.is_available() else "cpu"


class CycleGANTrainer:
    def __init__(self, model: models.CycleGAN, dataloader_a, dataloader_b, main_folder):
        self.model = model
        self.dataloader_a = dataloader_a
        self.dataloader_b = dataloader_b
        self.main_folder = main_folder
        self.losses_file = f"{main_folder}/losses.csv"

        if os.path.exists(self.losses_file):
            os.remove(self.losses_file)

        losses_file = open(self.losses_file, "x")
        losses_file.write("loss_gen_a,loss_gen_b,loss_des_a,loss_des_b\n")
        losses_file.close()

    def batch(self, batch_a, batch_b):
        self.model.set_input(batch_a, batch_b)
        self.model.optimize_parameters()

        losses_file = open(self.losses_file, "a")
        losses_file.write(f"{self.model.loss_gen_a},"
                          f"{self.model.loss_gen_b},"
                          f"{self.model.loss_des_a},"
                          f"{self.model.loss_des_b}\n")
        losses_file.close()

        return self.model.loss_gen_a, self.model.loss_gen_b, self.model.loss_des_a, self.model.loss_des_b

    def epoch(self, epoch: int):
        print(f"Epoch {epoch}")

        size = min(len(self.dataloader_a), len(self.dataloader_b))
        for i, (batch_a, batch_b) in enumerate(zip(self.dataloader_a, self.dataloader_b)):
            lga, lgb, lda, ldb = self.batch(
                batch_a[0].to(device),
                batch_b[0].to(device)
            )

            if i % 50 == 0:
                print(f"[{i:{len(str(size))}}/{size}] G_A:{lga:.5f} G_B:{lgb:.5f} D_A:{lda:.5f} D_B{ldb:.5f}")
                self.model.save(f"{self.main_folder}/epoch{epoch}")

        self.model.save(f"{self.main_folder}/epoch{epoch}")




