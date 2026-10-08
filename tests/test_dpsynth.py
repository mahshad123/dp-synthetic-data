import math

import numpy as np
import pandas as pd
import pytest

from dpsynth import Domain, IndependentSynthesizer, MSTSynthesizer, PrivacyLedger
from dpsynth.accounting import eps_delta_to_rho, gaussian_sigma, rho_to_eps_delta
from dpsynth.datasets import CENSUS_SCHEMA, make_census_like
from dpsynth.evaluate import marginal_tvd, ml_efficacy
from dpsynth.mechanisms import exponential_mechanism, marginal, project_to_simplex


@pytest.fixture(scope="module")
def data():
    return make_census_like(8_000, seed=0), Domain.from_dict(CENSUS_SCHEMA)


# ---------------- accounting ----------------
@pytest.mark.parametrize("eps", [0.1, 1.0, 5.0])
@pytest.mark.parametrize("delta", [1e-5, 1e-9])
def test_rho_conversion_roundtrip(eps, delta):
    assert math.isclose(rho_to_eps_delta(eps_delta_to_rho(eps, delta), delta), eps, rel_tol=1e-9)


def test_ledger_blocks_overspend():
    ledger = PrivacyLedger(total_rho=1.0)
    ledger.spend(0.6, "a")
    with pytest.raises(RuntimeError):
        ledger.spend(0.5, "b")


def test_gaussian_sigma_matches_definition():
    rho = 0.02
    assert math.isclose(1.0 / (2 * gaussian_sigma(rho) ** 2), rho)


@pytest.mark.parametrize("cls", [IndependentSynthesizer, MSTSynthesizer])
def test_synthesizer_spends_exactly_its_budget(cls, data):
    df, dom = data
    s = cls(epsilon=1.0, delta=1e-6, seed=0).fit(df, dom)
    assert math.isclose(s.epsilon_spent, 1.0, rel_tol=1e-6)


# ---------------- mechanisms ----------------
def test_marginal_counts(data):
    df, dom = data
    m = marginal(dom.encode(df), dom.shape, (1, 2))
    assert m.shape == (5, 6) and m.sum() == len(df)


def test_projection_on_simplex():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 5, size=50)
    y = project_to_simplex(x, 100.0)
    assert (y >= 0).all() and math.isclose(y.sum(), 100.0)
    # Already-feasible points are fixed points of the projection.
    z = np.abs(x); z = z / z.sum() * 10
    np.testing.assert_allclose(project_to_simplex(z, 10.0), z)


def test_exponential_mechanism_prefers_high_scores():
    rng = np.random.default_rng(0)
    picks = [exponential_mechanism(np.array([0.0, 0.0, 10.0]), 2.0, 1.0, rng) for _ in range(2000)]
    assert np.mean(np.array(picks) == 2) > 0.99


# ---------------- synthesizers ----------------
@pytest.mark.parametrize("cls", [IndependentSynthesizer, MSTSynthesizer])
def test_output_respects_schema(cls, data):
    df, dom = data
    out = cls(epsilon=1.0, seed=1).fit(df, dom).sample(500)
    assert list(out.columns) == dom.names and len(out) == 500
    assert set(out["education"]).issubset(CENSUS_SCHEMA["education"]["categories"])
    assert out["age"].between(17, 90).all()


def test_mst_learns_a_spanning_tree(data):
    df, dom = data
    s = MSTSynthesizer(epsilon=2.0, seed=0).fit(df, dom)
    assert len(s.tree_edges) == len(dom) - 1
    nodes = {n for e in s.tree_edges for n in e}
    assert nodes == set(range(len(dom)))


def test_mst_recovers_strongest_dependency(data):
    df, dom = data
    s = MSTSynthesizer(epsilon=5.0, seed=0).fit(df, dom)
    assert "education -- occupation" in s.describe_tree() or \
        "occupation -- education" in s.describe_tree()


def test_mst_beats_independent_on_pairwise_structure(data):
    df, dom = data
    mst = MSTSynthesizer(epsilon=2.0, seed=0).fit(df, dom).sample()
    ind = IndependentSynthesizer(epsilon=2.0, seed=0).fit(df, dom).sample()
    assert marginal_tvd(df, mst, dom, 2) < marginal_tvd(df, ind, dom, 2)


def test_seed_is_reproducible(data):
    df, dom = data
    a = MSTSynthesizer(epsilon=1.0, seed=42).fit(df, dom).sample(300)
    b = MSTSynthesizer(epsilon=1.0, seed=42).fit(df, dom).sample(300)
    pd.testing.assert_frame_equal(a, b)


def test_sample_before_fit_raises():
    with pytest.raises(RuntimeError):
        MSTSynthesizer().sample(10)


def test_tstr_signal_preserved(data):
    df, dom = data
    test = make_census_like(3_000, seed=99)
    synth = MSTSynthesizer(epsilon=4.0, seed=0).fit(df, dom).sample()
    assert ml_efficacy(synth, test, dom, "income", ">50K")["auc"] > 0.65
