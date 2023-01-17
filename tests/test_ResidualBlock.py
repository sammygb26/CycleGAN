import random

import torch
from cyclegan.models import ResidualBlock


def test_tensor_sizes():
    rb = ResidualBlock(3)

    size = random.randint(1, 100)

    start = torch.empty(3, size, size)
    result = rb(start)

    print(f"start : {str(start.size())}\nresult : {str(result.size())}")

    assert start.size() == result.size()
