from datetime import date

import pytest

from openrouter_models import (
    ModelInfo,
    estimate_cost,
    filter_models,
    format_label,
    format_usd,
    required_context,
    tokens_cost,
)

TODAY = date(2026, 9, 27)


def raw(
    id,
    prompt="0.0000001",
    completion="0.0000004",
    context=128000,
    inputs=("text",),
    outputs=("text",),
    params=("structured_outputs", "tools"),
    expiration=None,
    tokenizer="GPT",
):
    return {
        "id": id,
        "name": id.upper(),
        "context_length": context,
        "architecture": {
            "input_modalities": list(inputs),
            "output_modalities": list(outputs),
            "tokenizer": tokenizer,
        },
        "pricing": {"prompt": prompt, "completion": completion},
        "supported_parameters": list(params),
        "expiration_date": expiration,
    }


def ids(models):
    return [m.id for m in models]


def run(models, **kwargs):
    opts = dict(min_context=10000, max_input_price=1.0, include_free=False, today=TODAY)
    opts.update(kwargs)
    return filter_models(models, **opts)


def test_keeps_a_normal_model():
    assert ids(run([raw("a/ok")])) == ["a/ok"]


@pytest.mark.parametrize(
    "model",
    [
        raw("router/auto", prompt="-1", completion="-1"),
        raw("openrouter/free", prompt="0", completion="0", tokenizer="Router"),
        raw("a/bad-price", prompt="n/a"),
        raw("a/model:batch"),
        raw("~a/model-latest"),
        raw("a/image-gen", outputs=("image", "text")),
        raw("a/speech", outputs=("text", "audio")),
        raw("a/image-only-in", inputs=("image",)),
        raw("a/no-structured", params=("temperature",)),
        raw("a/expired", expiration="2026-09-01"),
        raw("a/tiny", context=4000),
    ],
    ids=lambda m: m["id"],
)
def test_excludes_unsuitable_models(model):
    assert run([model]) == []


def test_keeps_model_with_tools_only_and_future_expiration():
    models = [raw("a/tools", params=("tools",), expiration="2026-11-11")]
    assert ids(run(models)) == ["a/tools"]


def test_free_models_hidden_unless_included():
    models = [raw("a/m:free", prompt="0", completion="0"), raw("a/paid")]
    assert ids(run(models)) == ["a/paid"]
    assert ids(run(models, include_free=True)) == ["a/m:free", "a/paid"]


def test_zero_priced_models_count_as_free():
    models = [raw("stealth/preview", prompt="0", completion="0")]
    assert run(models) == []
    (m,) = run(models, include_free=True)
    assert m.is_free


def test_price_cap_is_per_million_input_tokens():
    models = [raw("a/cheap", prompt="0.000001"), raw("a/pricey", prompt="0.0000011")]
    assert ids(run(models, max_input_price=1.0)) == ["a/cheap"]


def test_sorted_by_input_then_output_price_then_id():
    models = [
        raw("c/x", prompt="0.0000002", completion="0.000001"),
        raw("b/x", prompt="0.0000001", completion="0.000002"),
        raw("a/x", prompt="0.0000001", completion="0.000002"),
        raw("d/x", prompt="0.0000001", completion="0.000001"),
    ]
    assert ids(run(models)) == ["d/x", "a/x", "b/x", "c/x"]


def test_parsed_model_info():
    (m,) = run([raw("a/ok", prompt="0.0000003", completion="0.0000025")])
    assert m.name == "A/OK"
    assert m.context_length == 128000
    assert m.input_per_million == pytest.approx(0.3)
    assert m.output_per_million == pytest.approx(2.5)
    assert not m.is_free


def test_required_context():
    assert required_context(20000) == 5000 + 450 + 2000


def test_estimate_cost_truncates_chapters_to_limit():
    m = ModelInfo("a/x", "X", 100000, prompt_price=1e-6, completion_price=2e-6)
    # chapter 1: 4000 chars -> 1000 + 450 input tokens; chapter 2: capped at 8000 -> 2000 + 450
    expected_input = (1450 + 2450) * 1e-6
    expected_output = 2 * 700 * 2e-6
    assert estimate_cost(m, [4000, 50000], 8000) == pytest.approx(expected_input + expected_output)


def test_tokens_cost():
    m = ModelInfo("a/x", "X", 100000, prompt_price=1e-6, completion_price=2e-6)
    assert tokens_cost(m, 1000, 500) == pytest.approx(0.002)


@pytest.mark.parametrize(
    "amount, text",
    [(0, "$0.00"), (0.0042, "$0.0042"), (0.04, "$0.04"), (1.234, "$1.23"), (12.5, "$12.50")],
)
def test_format_usd(amount, text):
    assert format_usd(amount) == text


def test_format_label():
    m = ModelInfo("a/x", "X", 100000, prompt_price=3e-7, completion_price=2.5e-6)
    assert format_label(m, 0.04) == "a/x — $0.30 in / $2.50 out per M — est. $0.04"
    free = ModelInfo("a/x:free", "X", 100000, prompt_price=0, completion_price=0)
    assert format_label(free, 0) == "a/x:free — free (rate-limited)"
