import random

import torch

from dalm.training.utils.train_utils import (
    get_cosine_sim,
    get_nt_xent_loss,
)


# TODO: look into using the `hypothesis` library for property-based testing
def test_get_cosine_sim() -> None:
    batch_size = random.randint(1, 10)

    a = torch.rand(10)
    a = a / torch.norm(a)
    a = torch.stack([a for _ in range(batch_size)])

    assert torch.allclose(get_cosine_sim(a, a, 1), torch.ones(batch_size, batch_size))

    b = -a

    assert torch.allclose(get_cosine_sim(a, b, 1), -torch.ones(batch_size, batch_size))

    cosine = random.uniform(-1, 1)

    b = cosine * a

    random_scale = random.uniform(1, 10)

    assert torch.allclose(
        get_cosine_sim(a, b, random_scale), random_scale * torch.ones(batch_size, batch_size) * cosine
    )


def test_get_nt_xent_loss() -> None:
    # generate a random diagonal matrix the log of whose is [1,2 , 3 .. n] and every other element is negative infinity
    square_size = random.randint(2, 10)
    a = torch.ones(square_size, square_size) * -float("inf")
    a.diagonal().copy_(torch.arange(1, square_size + 1, dtype=torch.float))

    print(get_nt_xent_loss(a))

    assert torch.allclose(get_nt_xent_loss(a), torch.tensor(0.0))

    # generate a random vector and stack it vertically
    a = torch.log(torch.rand(square_size))
    a = torch.stack([a for _ in range(len(a))])
    a = a.t()

    assert torch.allclose(get_nt_xent_loss(a), -torch.log(torch.tensor(1 / square_size)))


def test_get_nll() -> None:
    assert True
