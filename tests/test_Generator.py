import torch
from cyclegan import models


def test_tensor_sizes():
    nc_in = 3
    nc_out = 1
    size = 128

    g1 = models.Generator(nc_in, nc_out)
    g2 = models.Generator(nc_out, nc_in)

    before = torch.empty(nc_in, size, size)
    after = g1(before)
    reset = g2(after)

    assert before.size()[1:] == after.size()[1:]
    assert before.size()[0] == nc_in
    assert after.size()[0] == nc_out
    assert reset.size() == before.size()
