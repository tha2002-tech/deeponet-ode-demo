import torch

from deeponet_ode.model import DeepONet, make_branch_input, relative_l2


def test_forward_shape_and_finite():
    model = DeepONet()
    f = torch.randn(4, 100)
    k = torch.rand(4)
    t = torch.linspace(0, 1, 100).view(1, 100, 1).expand(4, 100, 1)
    out = model(make_branch_input(f, k), t)
    assert out.shape == (4, 100)
    assert torch.isfinite(out).all()


def test_make_branch_input_puts_k_last():
    f = torch.zeros(3, 5)
    k = torch.tensor([1.0, 2.0, 3.0])
    x = make_branch_input(f, k)
    assert x.shape == (3, 6)
    assert torch.equal(x[:, -1], k)


def test_branch_stats_are_saved_in_state_dict_and_applied():
    model = DeepONet(n_sensors=5)
    mean = torch.arange(6.0)
    std = torch.full((6,), 2.0)
    model.set_branch_stats(mean, std)
    sd = model.state_dict()
    assert torch.equal(sd["branch_mean"], mean)
    assert torch.equal(sd["branch_std"], std)
    fresh = DeepONet(n_sensors=5)
    fresh.load_state_dict(sd)
    x = torch.randn(2, 6)
    t = torch.rand(2, 3, 1)
    assert torch.allclose(fresh(x, t), model(x, t))


def test_relative_l2_values():
    a = torch.tensor([[3.0, 4.0]])
    assert relative_l2(a, a).item() == 0.0
    assert abs(relative_l2(torch.zeros(1, 2), a).item() - 1.0) < 1e-6


def test_short_training_reduces_loss():
    torch.manual_seed(0)
    model = DeepONet(n_sensors=10, width=32, p=16)
    f = torch.randn(16, 10)
    k = torch.rand(16)
    t = torch.linspace(0, 1, 20).view(1, 20, 1).expand(16, 20, 1)
    target = torch.sin(3 * t[..., 0]) * f[:, :1]
    x = make_branch_input(f, k)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    first = None
    for _ in range(200):
        opt.zero_grad()
        loss = torch.mean((model(x, t) - target) ** 2)
        loss.backward()
        opt.step()
        first = loss.item() if first is None else first
    assert loss.item() < 0.5 * first
