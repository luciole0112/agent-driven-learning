"""Tests for the MobileNetV2 / CIFAR-10 environment (offline, CIFAR-shaped stand-in data)."""

import numpy as np
import pytest
import torch

from adl.baselines import mbv2_methods as M
from adl.data.cifar10 import fake_cifar
from adl.env.budget import Budget
from adl.env.mbv2_env import TOOL_SPECS, MBV2Environment, MBV2Guard
from adl.models.mobilenetv2 import MBV2

torch.set_num_threads(4)


@pytest.fixture(scope="module")
def splits():
    return fake_cifar(260, 40, seed=1)


def make_env(splits, **kw):
    m = MBV2.random_init(0)
    M.recalibrate_bn(m, splits["train"]["x"], 128)
    kw.setdefault("budget", Budget())
    return MBV2Environment(m, splits, pool_n=120, retention_n=40, batch_size=64, **kw)


def logits(m, x):
    with torch.no_grad():
        return m(torch.from_numpy(x)).numpy()


def test_catalogue_and_specs():
    m = MBV2.random_init(0)
    names = m.layer_names()
    assert names[0] == "stem" and names[-1] == "head" and "b1.expand" not in names
    assert m.n_params() == 2_236_682
    assert m.layer_info("b4.project").residual is False and m.layer_info("b5.project").residual is True
    for k in m.named_tensors():
        assert k.split(".")[0] in {"stem", "last", "head"} or k[0] == "b"
    env_methods = {n[5:] for n in dir(MBV2Environment) if n.startswith("tool_")}
    assert set(TOOL_SPECS) == env_methods


def test_folded_kernel_matches_conv_bn():
    m = MBV2.random_init(0)
    bn = m.modules_of("stem")["bn"]
    bn.running_mean.uniform_(-1, 1)
    bn.running_var.uniform_(0.5, 2)
    x = torch.randn(2, 3, 8, 8)
    w, b = m.folded_kernel("stem")
    conv = m.modules_of("stem")["conv"]
    ref = bn.eval()(conv(x))
    out = torch.nn.functional.conv2d(x, w, b, stride=conv.stride, padding=conv.padding)
    assert torch.allclose(ref, out, atol=1e-5)


def test_structure_edits_preserve_function(splits):
    m = MBV2.random_init(0)
    M.recalibrate_bn(m, splits["train"]["x"], 64)
    x = splits["dev"]["x"][:8]
    y0 = logits(m, x)
    c = m.add_hidden_channel(4, torch.randn(24), torch.randn(3, 3), torch.zeros(32))
    assert c == 144 and m.channels("b4.expand") == 145 and m.channels("b4.dw") == 145
    assert np.allclose(logits(m, x), y0, atol=1e-5)
    m.add_feature_channel(torch.randn(320), torch.zeros(10))
    assert m.channels("last") == 1281
    assert np.allclose(logits(m, x), y0, atol=1e-5)
    m.remove_hidden_channel(4, 144)
    assert m.channels("b4.expand") == 144
    assert np.allclose(logits(m, x), y0, atol=1e-5)


def test_tools_run_without_gradients_and_probe_is_temporary(splits):
    env = make_env(splits)
    before = logits(env.model, splits["dev"]["x"][:8])
    calls = [("inspect_model", {}), ("confusion", {}), ("show_examples", {"cls": 2, "n": 4}),
             ("render_kernels", {"layer": "stem"}), ("render_kernels", {"layer": "b3.dw", "channels": [0, 1]}),
             ("render_kernels", {"layer": "b3.project", "channels": [0]}),
             ("channel_stats", {"layer": "b3.expand"}), ("channel_stats", {"layer": "b6.out", "cls_a": 3, "cls_b": 5}),
             ("top_activations", {"layer": "b4.dw", "channel": 1, "k": 4}),
             ("feature_map", {"layer": "stem", "channel": 0, "ids": [0, 1]}),
             ("probe", {"layer": "last", "channels": [0, 1, 2], "factor": 0.0}),
             ("capture_activations", {"layer": "b3.out", "key": "a"}),
             ("run_sandbox", {"code": "out['m'] = A['a'].mean(0)\nprint(A['a_y'].shape)"})]
    for name, args in calls:
        r = env.call(name, args)
        assert not r.text.startswith("ERROR"), (name, r.text)
    assert np.allclose(logits(env.model, splits["dev"]["x"][:8]), before)
    assert all(not p.requires_grad for p in env.model.parameters())
    assert env.budget.used["forward_images"] > 0


def test_sealed_splits_and_bad_args(splits):
    env = make_env(splits)
    assert env.call("evaluate", {"split": "heldout"}).text.startswith("ERROR")
    assert env.call("probe", {"layer": "b99.dw", "channels": [0]}).text.startswith("ERROR")
    assert env.call("scale_channel", {"layer": "stem", "channel": 99, "factor": 1.1}).text.startswith("ERROR")
    assert env.guard_events["invalid_args"] == 3


def test_budget_exhaustion_is_reported(splits):
    env = make_env(splits, budget=Budget(forward_images=100))
    r = env.call("evaluate", {"split": "dev"})        # dev has 50 images
    assert not r.text.startswith("ERROR")
    env.call("evaluate", {"split": "dev"})
    r = env.call("evaluate", {"split": "dev"})
    assert "exhausted" in r.text


def test_protocol_cycle_and_reject_restores(splits):
    env = make_env(splits, guard=MBV2Guard(max_acc_drop=None))
    w0 = env.model.named_tensors()["b2.dw.bn.weight"].clone()
    assert "BLOCKED" in env.call("scale_channel", {"layer": "b2.dw", "channel": 0, "factor": 1.5}).text
    assert "BLOCKED" in env.call("hypothesis", {"statement": "s", "evidence": "e", "edit_plan": "p",
                                                "prediction": "q"}).text
    env.call("show_examples", {"only_errors": True, "n": 4})
    env.call("hypothesis", {"statement": "s", "evidence": "e", "edit_plan": "p", "prediction": "q"})
    env.call("scale_channel", {"layer": "b2.dw", "channel": 0, "factor": 1.5})
    assert "BLOCKED" in env.call("submit").text
    env.call("evaluate", {"split": "dev"})
    env.call("decide", {"verdict": "reject", "reason": "test"})
    assert torch.equal(env.model.named_tensors()["b2.dw.bn.weight"], w0)
    assert env.cycles[0]["verdict"] == "reject"


def test_add_hidden_channel_tool_and_report(splits):
    env = make_env(splits)
    env.call("run_sandbox", {"code": "import numpy as np\nout['e']=np.ones(24)\nout['d']=np.ones((3,3))/9\n"
                                     "out['p']=np.zeros(32)"})
    r = env.call("add_hidden_channel", {"block": 4, "expand_key": "e", "dw_key": "d", "project_key": "p"})
    assert "added" in r.text
    assert env.call("recalibrate_bn", {"n": 64}).text.startswith("recalibrated")
    assert "fixed_rule:bn_recalibration" in env.ledger.flags()
    env.call("submit")
    rep = env.final_report()
    for k in ("acc_heldout_final", "acc_shift_final", "acc_shift_by_kind", "diff_from_initial"):
        assert k in rep
    assert "b4.expand.weight: shape [144, 24, 1, 1] -> [145, 24, 1, 1]" in rep["diff_from_initial"]


def test_baselines_respect_budget_and_copy(splits):
    m = MBV2.random_init(0)
    M.recalibrate_bn(m, splits["train"]["x"], 64)
    w0 = m.net.classifier[1].weight.clone()
    assert M.n_values(M.param_subset(m, "head")) == 12_810
    assert M.n_values(M.param_subset(m, "stem")) == 864 + 64
    m2, rep = M.es(m, splits, "head", 1500, pop=4, batch=32)
    assert rep["forward_images"] >= 1500 and rep["forward_images"] < 1500 + 4 * 32
    assert torch.equal(m.net.classifier[1].weight, w0)
    assert not torch.equal(m2.net.classifier[1].weight, w0)
    _, rep = M.random_search(m, splits, "head", 640, batch=32)
    assert rep["steps"] == 10
    m3, rep = M.backprop(m, splits, "head", 3 * 32 * 3, batch=32)
    assert rep["steps"] == 3 and all(p.grad is None and not p.requires_grad for p in m3.parameters())
    _, rep = M.ncm_head(m, splits)
    assert rep["space"] == "head"


def test_stimulus_and_readout_tools(splits):
    env = make_env(splits)
    r = env.call("stimulus_response", {"layer": "stem", "channel": 3})
    assert "orientation tuning" in r.text and r.images
    r = env.call("probe_readout", {"layer": "b3.out"})
    assert "dev accuracy" in r.text, r.text
    assert env.budget.used["evaluations"] == 1
