import json

import pytest
import torch

from adl.agent.loop import PROTOCOL_PROMPT, context_window, run_agent
from adl.data.shapes import ConceptSpec, make_splits
from adl.env.budget import Budget
from adl.env.environment import ADLEnvironment
from adl.env.guard import Guard
from adl.models.cnn import EditableCNN

HYP = {"statement": "label 1 has a diagonal bar", "positive_evidence": "bar at 45 deg",
       "negative_evidence": "no such bar", "edit_plan": "connect conv3[0] to head_concept",
       "prediction": "dev AUROC rises a little"}
EDIT = ("connect", {"src_layer": "conv3", "src_channel": 0, "dst": "head_concept",
                    "dst_channel": 0, "values": [0.5]})


@pytest.fixture(scope="module")
def splits():
    return make_splits(ConceptSpec(kind="glyph"), n_train=40, n_dev=40, n_heldout=40, n_shift=20)


def _env(splits, **kw):
    return ADLEnvironment(EditableCNN(), splits, Budget(), guard=Guard(**kw))


def _ready(env):
    env.call("show_examples", {"label": 1, "n": 2})
    env.call("show_examples", {"label": 0, "n": 2})
    assert "recorded" in env.call("hypothesis", HYP).text


def _same(a, b):
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def test_invalid_args_rejected_without_side_effects(splits):
    env = ADLEnvironment(EditableCNN(), splits, Budget())
    before = env.model.state_dict()
    bad = [
        ("connect", {"src_layer": "conv3", "src_channel": 20, "dst": "head_concept",
                     "dst_channel": 0, "values": [1, 1, 1, 1]}),
        ("connect", {"src_layer": "conv2", "src_channel": 5, "dst": "head_concept", "dst_channel": 0,
                     "values": [8]}),
        ("remove_channel", {"layer": "head_concept", "channel": 0}),
        ("scale_channel", {"layer": "conv3", "channel": 99, "factor": 1.5}),
        ("evaluate", {"split": "dev", "bogus": 1}),
        ("top_activations", {"layer": "conv3"}),
    ]
    for name, args in bad:
        assert env.call(name, args).text.startswith("ERROR (invalid arguments"), (name, args)
    assert _same(before, env.model.state_dict())
    assert env.budget.used["edits"] == 0 and env.guard_events["invalid_args"] == len(bad)
    # an integral float is accepted for an integer argument
    assert not env.call("scale_channel", {"layer": "conv1", "channel": 1.0, "factor": 1.0}).text.startswith("ERROR")


def test_scale_factor_bounded_in_guard_mode(splits):
    env = _env(splits)
    _ready(env)
    assert "invalid arguments" in env.call("scale_channel", {"layer": "conv3", "channel": 3, "factor": 5}).text


def test_edit_needs_contrast_and_hypothesis(splits):
    env = _env(splits)
    assert "BLOCKED" in env.call(*EDIT).text
    env.call("show_examples", {"label": 1})
    assert "label=0" in env.call("hypothesis", HYP).text
    env.call("show_examples", {"label": 0})
    env.call("hypothesis", HYP)
    assert "BLOCKED" not in env.call(*EDIT).text


def test_cycle_requires_evaluate_then_decide(splits):
    env = _env(splits)
    _ready(env)
    before = env.model.state_dict()
    env.call(*EDIT)
    assert "evaluate" in env.call("scale_channel", {"layer": "conv1", "channel": 0, "factor": 1.5}).text
    assert "BLOCKED" in env.call("submit").text
    assert "BLOCKED" in env.call("decide", {"verdict": "accept", "reason": "x"}).text
    r = env.call("evaluate", {"split": "dev"})
    assert "decide(" in r.text
    assert "BLOCKED" in env.call(*EDIT).text
    env.call("decide", {"verdict": "reject", "reason": "no gain"})
    assert _same(before, env.model.state_dict())
    assert env.hypothesis is None and env.cycles[-1]["verdict"] == "reject"
    assert "BLOCKED" in env.call(*EDIT).text      # a new cycle needs a new hypothesis


def test_accept_keeps_edit(splits):
    env = _env(splits)
    _ready(env)
    env.call(*EDIT)
    env.call("evaluate", {})
    env.call("decide", {"verdict": "accept", "reason": "ok"})
    assert float(env.model.params["head_concept.weight"][0, 0]) == pytest.approx(0.5)
    rep = env.final_report()
    assert rep["cycle_verdicts"]["accept"] == 1 and not rep["unverified_edit_at_end"]
    json.dumps(rep)


def test_auto_rollback_on_base_accuracy_drop(splits):
    # a negative allowed drop makes any edit count as damaging
    env = _env(splits, max_base_drop=-1.0)
    _ready(env)
    before = env.model.state_dict()
    r = env.call("set_weights", {"name": "conv3.weight", "values": 0.0})
    assert "AUTO-ROLLBACK" in r.text and _same(before, env.model.state_dict())
    assert env.pending is None and env.hypothesis is not None
    assert env.final_report()["guard_events"]["auto_rollbacks"] == 1


def test_failed_edit_is_not_half_applied(splits):
    env = ADLEnvironment(EditableCNN(), splits, Budget())
    before = env.model.state_dict()
    r = env.call("set_weights", {"name": "conv1.weight", "index": [0], "values": [1.0, 2.0]})
    assert r.text.startswith("ERROR") and _same(before, env.model.state_dict())


def test_protocol_tools_only_with_guard(splits):
    assert "hypothesis" not in ADLEnvironment(EditableCNN(), splits, Budget()).available_tools()
    assert "decide" in _env(splits).available_tools()


def test_scripted_agent_full_cycle(splits):
    steps = [("show_examples", {"label": 1, "n": 2}), ("show_examples", {"label": 0, "n": 2}),
             ("hypothesis", HYP), EDIT, ("evaluate", {"split": "dev"}),
             ("decide", {"verdict": "accept", "reason": "test"}), ("submit", {})]
    seen = []

    class Scripted:
        def chat(self, messages, tools=None):
            seen.append(messages)
            name, args = steps.pop(0)
            return {"role": "assistant", "content": "", "tool_calls": [
                {"id": name, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]}

    env = _env(splits)
    rep = run_agent(env, Scripted(), log=lambda *_: None)
    assert env.submitted and rep["cycle_verdicts"]["accept"] == 1
    assert seen[0][0]["content"] == PROTOCOL_PROMPT
    assert seen[0][-1]["content"].startswith("[protocol]")
    assert "[protocol]" in context_window(seen[-1], env)[-1]["content"]
