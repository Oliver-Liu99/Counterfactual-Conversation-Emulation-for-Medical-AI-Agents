"""Tests for the trust-checklist diagnostic."""

from __future__ import annotations

from ccema.analysis.failure_modes import trust_report


def _kwargs(**override):
    base = dict(
        capability_gap=0.20,
        ess_per_n=0.18,
        classifier_auc=0.83,
        rubric_correlation=0.78,
        cce_vs_raw_auc_drop=0.09,
        conformal_ci_width=0.05,
        point_estimate=0.13,
        msm_sign_robust_at_gamma2=True,
        process_reward_stdev=0.09,
    )
    base.update(override)
    return base


def test_trust_report_all_pass_on_healthy_run() -> None:
    rep = trust_report(**_kwargs())
    assert rep.all_passed
    assert rep.n_passed == 8
    assert rep.n_total == 8


def test_trust_report_fails_when_capability_gap_low() -> None:
    rep = trust_report(**_kwargs(capability_gap=0.02))
    assert not rep.all_passed
    failing = [i for i in rep.items if not i.passed]
    assert any(i.name == "agent_capability" for i in failing)


def test_trust_report_fails_when_classifier_auc_high() -> None:
    rep = trust_report(**_kwargs(classifier_auc=0.97))
    failing = [i for i in rep.items if not i.passed]
    assert any(i.name == "positivity_auc" for i in failing)


def test_trust_report_fails_when_msm_not_robust() -> None:
    rep = trust_report(**_kwargs(msm_sign_robust_at_gamma2=False))
    failing = [i for i in rep.items if not i.passed]
    assert any(i.name == "msm_robust" for i in failing)


def test_trust_report_fails_when_ci_too_wide() -> None:
    rep = trust_report(**_kwargs(conformal_ci_width=2.0, point_estimate=0.1))
    failing = [i for i in rep.items if not i.passed]
    assert any(i.name == "conformal_ci_width" for i in failing)


def test_trust_report_fails_when_process_dispersion_low() -> None:
    rep = trust_report(**_kwargs(process_reward_stdev=0.01))
    failing = [i for i in rep.items if not i.passed]
    assert any(i.name == "process_reward_dispersion" for i in failing)


def test_trust_report_as_dict_serializable() -> None:
    import json

    rep = trust_report(**_kwargs())
    s = json.dumps(rep.as_dict())
    assert "items" in s
