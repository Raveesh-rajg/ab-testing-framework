import pytest

from expkit.sequential import msprt_pvalue, AlwaysValidTracker, PeekingSimulation


class TestMsprt:
    def test_no_effect_gives_high_p(self):
        p = msprt_pvalue(theta_hat=0.0, variance_of_estimate=1e-5, tau=0.005)
        assert p > 0.5

    def test_large_effect_gives_low_p(self):
        p = msprt_pvalue(theta_hat=0.02, variance_of_estimate=1e-6, tau=0.005)
        assert p < 0.001

    def test_p_capped_at_one(self):
        assert msprt_pvalue(0.0, 1.0, 0.001) <= 1.0

    def test_invalid_inputs_raise(self):
        with pytest.raises(ValueError):
            msprt_pvalue(0.1, -1.0, 0.01)
        with pytest.raises(ValueError):
            msprt_pvalue(0.1, 1.0, 0.0)


class TestTracker:
    def test_monotone_nonincreasing(self):
        tr = AlwaysValidTracker(tau=0.01)
        tr.update(0.001, 1e-5)
        tr.update(0.0, 1e-5)  # weaker evidence later must not raise p
        assert tr.history[1] <= tr.history[0]


class TestPeekingSimulation:
    """The headline claim of the module, verified empirically."""

    def test_naive_peeking_inflates_alpha_and_msprt_does_not(self):
        sim = PeekingSimulation(
            baseline=0.05, n_per_arm=10_000, n_looks=10, alpha=0.05, seed=7
        )
        result = sim.run(n_experiments=400)
        # Naive repeated testing should inflate well above nominal 5%...
        assert result.naive_false_positive_rate > 0.10
        # ...while mSPRT stays at or below it (it's conservative).
        assert result.msprt_false_positive_rate <= 0.05
