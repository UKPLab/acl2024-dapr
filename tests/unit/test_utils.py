from dapr.utils import Multiprocesser, hash_model


def func(x: int) -> str:
    return str(x + 1)


def test_multiprocesser() -> None:
    # Given:
    total = 100000

    # When:
    res = Multiprocesser(2).run(
        data=range(total), func=func, desc="Testing Multiprocesser", total=total
    )

    # Then:
    assert res == [str(i + 1) for i in range(total)]


def test_hash_model() -> None:
    # Given:
    import torch

    model1 = torch.nn.Transformer()
    model2 = torch.nn.Transformer()

    # When:
    # Then:
    assert hash_model(model1) == hash_model(model1)
    assert hash_model(model1) != hash_model(model2)
