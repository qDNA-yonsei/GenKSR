import pytest


torch = pytest.importorskip('torch')

from genksr.training.lr_schedulers import warm_up_cosine_lr_scheduler


def test_warmup_cosine_uses_absolute_final_learning_rate():
    parameter = torch.nn.Parameter(torch.tensor(0.0))
    optimizer = torch.optim.SGD([parameter], lr=1e-3)
    scheduler = warm_up_cosine_lr_scheduler(
        optimizer, epochs=10, warm_up_epochs=2, eta_min=1e-7)

    learning_rates = [optimizer.param_groups[0]['lr']]
    for _ in range(10):
        optimizer.step()
        scheduler.step()
        learning_rates.append(optimizer.param_groups[0]['lr'])

    assert learning_rates[0] == pytest.approx(1e-7)
    assert learning_rates[2] == pytest.approx(1e-3)
    assert learning_rates[-1] == pytest.approx(1e-7)
    assert learning_rates[-1] != pytest.approx(1e-10)


def test_warmup_cosine_rejects_invalid_schedule():
    parameter = torch.nn.Parameter(torch.tensor(0.0))
    optimizer = torch.optim.SGD([parameter], lr=1e-3)

    with pytest.raises(ValueError):
        warm_up_cosine_lr_scheduler(
            optimizer, epochs=10, warm_up_epochs=10, eta_min=1e-7)
