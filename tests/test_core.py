import numpy as np
import pytest
import torch

from adl.agent.loop import run_agent, tool_schemas
from adl.baselines import methods as B
from adl.data.shapes import ConceptSpec, make_dataset, make_splits
from adl.env.budget import Budget, BudgetExceeded
from adl.env.environment import ADLEnvironment, TOOL_SPECS
from adl.env.metrics import auroc
from adl.env.sandbox import SandboxViolation, run_code, scan_code
from adl.models.cnn import EditableCNN


@pytest.fixture(scope="module")
def splits():
    return make_splits(ConceptSpec(kind="glyph"), n_train=40, n_dev=40, n_heldout=40, n_shift=20)


def test_dataset_deterministic():
    a = make_dataset(5, 3, ConceptSpec())
    b = make_dataset(5, 3, ConceptSpec())
    assert np.array_equal(a["x"], b["x"]) and a["x"].shape == (5, 3, 64, 64)


def test_distractor_has_same_colour_counts():
    from adl.data.shapes import _distractor
    c = ConceptSpec().build()
    d = _distractor(c.glyph, np.random.default_rng(0))
    assert sorted(d.flatten()) == sorted(c.glyph.flatten()) and not np.array_equal(d, c.glyph)


def test_auroc():
    assert auroc(np.array([0.1, 0.2, 0.8, 0.9]), np.array([0, 0, 1, 1])) == 1.0
    assert auroc(np.zeros(4), np.array([0, 1, 0, 1])) == 0.5


def test_add_channel_preserves_function():
    m = EditableCNN()
    x = torch.rand(2, 3, 64, 64)
    b0, c0 = m(x)
    for layer in ("conv1", "conv2", "conv3"):
        m.add_channel(layer, torch.randn(m.params[f"{layer}.weight"].shape[1], *m.params[f"{layer}.weight"].shape[2:]), 0.3)
    b1, c1 = m(x)
    assert torch.allclose(b0, b1, atol=1e-5) and torch.allclose(c0, c1, atol=1e-5)
    m.remove_channel("conv3", m.channels("conv3") - 1)
    assert m.params["head_base.weight"].shape[1] == m.channels("conv3")


def test_env_no_grad_and_sealed(splits):
    env = ADLEnvironment(EditableCNN(), splits, Budget())
    r = env.call("evaluate", {"split": "heldout"})
    assert r.text.startswith("ERROR")
    env.call("add_channel", {"layer": "conv3"})
    env.call("connect", {"src_layer": "conv3", "src_channel": 32, "dst": "head_concept",
                         "dst_channel": 0, "values": 1.0})
    for v in env.model.params.values():
        assert not v.requires_grad
    rep = env.final_report()
    assert "auroc_heldout" in rep


def test_rollback(splits):
    env = ADLEnvironment(EditableCNN(), splits, Budget())
    before = env.model.state_dict()
    env.call("scale_channel", {"layer": "conv1", "channel": 0, "factor": 3.0})
    env.call("rollback", {"tag": "initial"})
    assert all(torch.equal(before[k], env.model.params[k]) for k in before)


def test_budget_enforced(splits):
    env = ADLEnvironment(EditableCNN(), splits, Budget(evaluations=1))
    assert "concept AUROC" in env.call("evaluate").text
    assert "exhausted" in env.call("evaluate").text


def test_sandbox_flags_and_blocks():
    assert "fit:least_squares" in scan_code("x = np.linalg.lstsq(a, b)")
    assert "suspect:iterative_update_loop" in scan_code("for i in range(3):\n    w -= 0.1 * g")
    with pytest.raises(SandboxViolation):
        run_code("import torch", lambda n: None, {})
    with pytest.raises(SandboxViolation):
        run_code("np.linalg.pinv(np.eye(2))", lambda n: None, {}, forbid_fitting=True)
    text, out, _ = run_code("out['k'] = np.ones(3); print('hi')", lambda n: None, {})
    assert "hi" in text and out["k"].shape == (3,)


def test_conditions_restrict_tools(splits):
    env = ADLEnvironment(EditableCNN(), splits, Budget(), condition="no_observation")
    assert "top_activations" not in env.available_tools()
    assert env.call("top_activations", {"layer": "conv1", "channel": 0}).text.startswith("ERROR")
    env2 = ADLEnvironment(EditableCNN(), splits, Budget(), condition="no_semantics")
    assert env2.call("show_examples", {}).images == []


def test_tool_specs_cover_methods():
    for name in TOOL_SPECS:
        assert hasattr(ADLEnvironment, f"tool_{name}"), name


def test_agent_loop_with_scripted_llm(splits):
    class Scripted:
        def __init__(self):
            self.steps = [("inspect_model", {}), ("show_examples", {"n": 2}),
                          ("add_channel", {"layer": "conv3"}), ("evaluate", {}), ("submit", {})]

        def chat(self, messages, tools=None):
            name, args = self.steps.pop(0)
            return {"role": "assistant", "content": "", "tool_calls": [
                {"id": name, "type": "function", "function": {"name": name, "arguments": __import__("json").dumps(args)}}]}

    env = ADLEnvironment(EditableCNN(), splits, Budget())
    assert len(tool_schemas(env)) == len(env.available_tools())
    rep = run_agent(env, Scripted(), log=lambda *_: None)
    assert env.submitted and "auroc_heldout" in rep


def test_es_respects_budget(splits):
    b = Budget(forward_passes=10)
    B.evolution_strategy(EditableCNN(), splits["dev"], b)
    assert b.used["forward_passes"] <= 10
    with pytest.raises(BudgetExceeded):
        b.charge("forward_passes", 1)


def test_context_window_keeps_pairs_and_drops_old_images(splits):
    from adl.agent.loop import context_window
    env = ADLEnvironment(EditableCNN(), splits, Budget())
    env.notes.append("n1")
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "t"}]
    for i in range(40):
        msgs += [{"role": "assistant", "content": "", "tool_calls": [{"id": str(i)}]},
                 {"role": "tool", "tool_call_id": str(i), "content": "r"},
                 {"role": "user", "content": [{"type": "text", "text": "img"}]}]
    view = context_window(msgs, env, keep_last=10, keep_images=1)
    assert view[0]["role"] == "system" and "n1" in view[2]["content"]
    assert view[3]["role"] == "assistant"
    assert sum(isinstance(m.get("content"), list) for m in view) == 1
