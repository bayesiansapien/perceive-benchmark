"""
Pytest integration tests for src/sampling/structural_prior.py

Coverage:
  - VDS estimation rules
  - RDS estimation rules
  - SES estimation rules
  - Dataset-level hard priors (RVL-CDIP, MP-DocVQA, SlideVQA, CORD/SROIE, TabFact, WTQ)
  - composite_est formula: 0.30*VDS + 0.45*RDS + 0.25*SES
  - tier_prior_soft sums to ~1.0
  - All required output fields present
  - _compute_prior() is pure, no disk I/O
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from src.sampling.structural_prior import (
    _compute_prior,
    _estimate_vds,
    _estimate_rds,
    _estimate_ses,
    _compute_composite,
    _composite_to_tier,
    _tier_to_soft,
    _apply_dataset_priors,
    _TIER1_MAX,
    _TIER2_MAX,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

REQUIRED_FIELDS = {
    "vds_est",
    "rds_est",
    "ses_est",
    "composite_est",
    "tier_prior",
    "tier_prior_soft",
}


def _sample(
    query: str = "What is the title?",
    source_dataset: str = "DocVQA",
    num_pages: int = 1,
    has_chart: bool = False,
    has_figure: bool = False,
    has_table: bool = False,
) -> dict:
    """Return a minimal synthetic sample dict."""
    return {
        "sample_id": "test_001",
        "source_dataset": source_dataset,
        "query": query,
        "num_pages": num_pages,
        "has_chart": has_chart,
        "has_figure": has_figure,
        "has_table": has_table,
        "gt_answer": "Test answer",
        "task_type": "T1",
    }


# ===========================================================================
# VDS tests
# ===========================================================================

class TestEstimateVDS:
    """Visual Dependency Score estimation rules."""

    def test_vds_4_when_chart_and_figure(self):
        rec = _sample(has_chart=True, has_figure=True)
        assert _estimate_vds(rec) == 4

    def test_vds_3_when_chart_only(self):
        rec = _sample(has_chart=True, has_figure=False)
        assert _estimate_vds(rec) == 3

    def test_vds_3_when_figure_only(self):
        rec = _sample(has_chart=False, has_figure=True)
        assert _estimate_vds(rec) == 3

    def test_vds_2_when_spatial_keyword_present(self):
        rec = _sample(query="Where is the logo in the document?")
        assert _estimate_vds(rec) == 2

    def test_vds_2_spatial_keyword_case_insensitive(self):
        # "Position" → "position" after lower()
        rec = _sample(query="What is the Position of the stamp?")
        assert _estimate_vds(rec) == 2

    def test_vds_1_plain_query_no_visual(self):
        rec = _sample(query="What is the total amount?")
        assert _estimate_vds(rec) == 1

    def test_vds_4_overrides_spatial_keyword(self):
        # chart + figure wins regardless of query content
        rec = _sample(
            query="Where is the chart located?",
            has_chart=True,
            has_figure=True,
        )
        assert _estimate_vds(rec) == 4

    def test_vds_3_when_has_chart_true_regardless_of_query(self):
        rec = _sample(query="What is shown?", has_chart=True)
        assert _estimate_vds(rec) == 3


# ===========================================================================
# RDS tests
# ===========================================================================

class TestEstimateRDS:
    """Reasoning Depth Score estimation rules."""

    def test_rds_4_multi_doc_keyword_based_on(self):
        rec = _sample(query="Based on both documents, what changed?")
        assert _estimate_rds(rec) == 4

    def test_rds_4_cross_reference_keyword(self):
        rec = _sample(query="Cross-reference the figures across pages.")
        assert _estimate_rds(rec) == 4

    def test_rds_3_compare_keyword(self):
        rec = _sample(query="Compare the revenue figures for Q1 and Q2.")
        assert _estimate_rds(rec) == 3

    def test_rds_3_calculate_keyword(self):
        rec = _sample(query="Calculate the total from all line items.")
        assert _estimate_rds(rec) == 3

    def test_rds_3_sum_keyword(self):
        rec = _sample(query="What is the sum of all expenses listed?")
        assert _estimate_rds(rec) == 3

    def test_rds_3_difference_keyword(self):
        rec = _sample(query="What is the difference between the two values?")
        assert _estimate_rds(rec) == 3

    def test_rds_1_short_query_no_keywords(self):
        # fewer than 8 words, no calc/comparison keywords
        rec = _sample(query="What is the date?")
        assert _word_count("What is the date?") < 8
        assert _estimate_rds(rec) == 1

    def test_rds_2_long_query_no_keywords(self):
        # >= 8 words, no special keywords
        rec = _sample(query="What is the name of the company in the header section?")
        assert _estimate_rds(rec) == 2

    def test_rds_case_insensitive(self):
        # "Compare" should still match
        rec = _sample(query="Compare the two invoices please.")
        assert _estimate_rds(rec) == 3


def _word_count(q: str) -> int:
    return len(q.split())


# ===========================================================================
# SES tests
# ===========================================================================

class TestEstimateSES:
    """Spatial Extent Score estimation rules."""

    def test_ses_4_multi_page(self):
        rec = _sample(num_pages=3)
        assert _estimate_ses(rec) == 4

    def test_ses_4_exactly_2_pages(self):
        rec = _sample(num_pages=2)
        assert _estimate_ses(rec) == 4

    def test_ses_3_table_and_figure(self):
        rec = _sample(has_table=True, has_figure=True, num_pages=1)
        assert _estimate_ses(rec) == 3

    def test_ses_3_spatial_spread_keyword(self):
        rec = _sample(query="Across the entire document, find all occurrences.")
        assert _estimate_ses(rec) == 3

    def test_ses_3_span_keyword(self):
        rec = _sample(query="Information that span the multiple sections.")
        assert _estimate_ses(rec) == 3

    def test_ses_2_has_table_only(self):
        rec = _sample(has_table=True, has_figure=False, num_pages=1)
        assert _estimate_ses(rec) == 2

    def test_ses_2_has_figure_only(self):
        rec = _sample(has_table=False, has_figure=True, num_pages=1)
        assert _estimate_ses(rec) == 2

    def test_ses_1_plain_single_page(self):
        rec = _sample(num_pages=1)
        assert _estimate_ses(rec) == 1

    def test_ses_4_overrides_table_figure(self):
        # multi-page wins regardless of table/figure
        rec = _sample(has_table=True, has_figure=True, num_pages=5)
        assert _estimate_ses(rec) == 4


# ===========================================================================
# Composite formula test
# ===========================================================================

class TestCompositeFormula:
    """composite_est = 0.30*VDS + 0.45*RDS + 0.25*SES, rounded to 4dp."""

    @pytest.mark.parametrize("vds,rds,ses,expected", [
        (1, 1, 1, round(0.30 + 0.45 + 0.25, 4)),   # = 1.0
        (4, 4, 4, round(1.20 + 1.80 + 1.00, 4)),   # = 4.0
        (2, 3, 2, round(0.60 + 1.35 + 0.50, 4)),   # = 2.45
        (1, 4, 4, round(0.30 + 1.80 + 1.00, 4)),   # = 3.1
        (3, 2, 1, round(0.90 + 0.90 + 0.25, 4)),   # = 2.05
    ])
    def test_composite_formula(self, vds, rds, ses, expected):
        assert _compute_composite(vds, rds, ses) == pytest.approx(expected, abs=1e-4)

    def test_compute_prior_composite_matches_formula(self):
        rec = _sample(
            query="Where is the figure located across the document?",
            has_chart=True,
            has_figure=True,
            num_pages=2,
        )
        prior = _compute_prior(rec)
        expected = _compute_composite(prior["vds_est"], prior["rds_est"], prior["ses_est"])
        assert prior["composite_est"] == pytest.approx(expected, abs=1e-4)


# ===========================================================================
# tier_prior_soft validity
# ===========================================================================

class TestTierPriorSoft:
    """tier_prior_soft must be a 3-element list summing to 1.0."""

    @pytest.mark.parametrize("tier,composite", [
        (1, 1.0),
        (1, 2.1),
        (2, 2.5),
        (2, 3.0),
        (3, 3.5),
        (3, 4.0),
    ])
    def test_soft_sums_to_one(self, tier, composite):
        soft = _tier_to_soft(tier, composite)
        assert len(soft) == 3
        assert sum(soft) == pytest.approx(1.0, abs=1e-4)

    def test_soft_all_elements_in_unit_interval(self):
        for tier in (1, 2, 3):
            soft = _tier_to_soft(tier, 2.5)
            for p in soft:
                assert 0.0 <= p <= 1.0

    def test_compute_prior_soft_sums_to_one(self):
        rec = _sample(query="What is the total?")
        prior = _compute_prior(rec)
        soft = prior["tier_prior_soft"]
        assert len(soft) == 3
        assert sum(soft) == pytest.approx(1.0, abs=1e-4)

    def test_tier1_dominant_probability_for_easy_sample(self):
        # A plain single-page, no visual, short query → VDS=1, RDS=1, SES=1
        rec = _sample(query="What is the title?")
        prior = _compute_prior(rec)
        soft = prior["tier_prior_soft"]
        assert soft[0] > soft[1], "P(T1) should exceed P(T2) for easy sample"
        assert soft[0] > soft[2], "P(T1) should exceed P(T3) for easy sample"

    def test_tier3_dominant_probability_for_hard_sample(self):
        # chart + figure, multi-page, cross-reference query → VDS=4, RDS=4, SES=4
        rec = _sample(
            query="Cross-reference the figures across pages to calculate the total.",
            has_chart=True,
            has_figure=True,
            num_pages=3,
        )
        prior = _compute_prior(rec)
        soft = prior["tier_prior_soft"]
        assert soft[2] > soft[0], "P(T3) should exceed P(T1) for hard sample"
        assert soft[2] > soft[1], "P(T3) should exceed P(T2) for hard sample"


# ===========================================================================
# Required output fields
# ===========================================================================

class TestRequiredFields:
    """_compute_prior() must return all required keys."""

    def test_all_required_fields_present(self):
        rec = _sample()
        prior = _compute_prior(rec)
        assert REQUIRED_FIELDS.issubset(prior.keys()), (
            f"Missing fields: {REQUIRED_FIELDS - prior.keys()}"
        )

    def test_vds_est_in_range(self):
        rec = _sample(has_chart=True)
        prior = _compute_prior(rec)
        assert prior["vds_est"] in (1, 2, 3, 4)

    def test_rds_est_in_range(self):
        rec = _sample(query="Compare Q1 and Q2 totals.")
        prior = _compute_prior(rec)
        assert prior["rds_est"] in (1, 2, 3, 4)

    def test_ses_est_in_range(self):
        rec = _sample(num_pages=2)
        prior = _compute_prior(rec)
        assert prior["ses_est"] in (1, 2, 3, 4)

    def test_tier_prior_in_range(self):
        rec = _sample()
        prior = _compute_prior(rec)
        assert prior["tier_prior"] in (1, 2, 3)

    def test_composite_est_is_float(self):
        rec = _sample()
        prior = _compute_prior(rec)
        assert isinstance(prior["composite_est"], float)

    def test_tier_prior_soft_is_list_of_three(self):
        rec = _sample()
        prior = _compute_prior(rec)
        assert isinstance(prior["tier_prior_soft"], list)
        assert len(prior["tier_prior_soft"]) == 3


# ===========================================================================
# Dataset-level hard priors
# ===========================================================================

class TestDatasetPriors:
    """Dataset-specific overrides applied by _apply_dataset_priors."""

    # ── RVL-CDIP ────────────────────────────────────────────────────────────────

    def test_rvlcdip_tier_prior_always_1(self):
        # Regardless of query complexity, RVL-CDIP must produce tier_prior == 1
        rec = _sample(
            source_dataset="RVL-CDIP",
            query="Cross-reference figures across pages to calculate total.",
            has_chart=True,
            has_figure=True,
            num_pages=3,
        )
        prior = _compute_prior(rec)
        assert prior["tier_prior"] == 1, (
            f"RVL-CDIP should always have tier_prior=1, got {prior['tier_prior']}"
        )

    def test_rvlcdip_case_variants(self):
        """Source dataset normalisation: 'rvl-cdip' and 'rvlcdip' both apply prior."""
        for ds_name in ("rvl-cdip", "rvlcdip", "RVL-CDIP"):
            rec = _sample(source_dataset=ds_name, num_pages=5, has_chart=True, has_figure=True)
            prior = _compute_prior(rec)
            assert prior["tier_prior"] == 1, f"Dataset '{ds_name}' should give tier_prior=1"

    # ── MP-DocVQA ────────────────────────────────────────────────────────────────

    def test_mpdocvqa_ses_always_4(self):
        # MP-DocVQA must have ses_est == 4
        rec = _sample(
            source_dataset="MP-DocVQA",
            num_pages=1,    # single page in sample, prior should still set ses=4
        )
        prior = _compute_prior(rec)
        assert prior["ses_est"] == 4, (
            f"MP-DocVQA should always have ses_est=4, got {prior['ses_est']}"
        )

    def test_mpdocvqa_tier_not_1(self):
        # MP-DocVQA cannot be Tier 1 (multi-page doc implies at least Tier 2)
        rec = _sample(source_dataset="MP-DocVQA", query="What is the title?")
        prior = _compute_prior(rec)
        assert prior["tier_prior"] >= 2, (
            f"MP-DocVQA tier_prior should be >= 2, got {prior['tier_prior']}"
        )

    def test_mpdocvqa_ses4_flows_into_composite(self):
        # With ses_est=4, composite must include 0.25*4 = 1.0 for ses component
        rec = _sample(source_dataset="MP-DocVQA", query="What is the title?")
        prior = _compute_prior(rec)
        vds = prior["vds_est"]
        rds = prior["rds_est"]
        ses = prior["ses_est"]
        expected = _compute_composite(vds, rds, ses)
        assert prior["composite_est"] == pytest.approx(expected, abs=1e-4)
        # ses must contribute 1.0 (= 0.25 * 4)
        assert ses == 4

    # ── SlideVQA ────────────────────────────────────────────────────────────────

    def test_slidevqa_arithmetic_query_tier_at_least_2(self):
        rec = _sample(
            source_dataset="SlideVQA",
            query="Calculate the percentage increase between the two slides.",
        )
        prior = _compute_prior(rec)
        assert prior["tier_prior"] >= 2

    def test_slidevqa_non_arithmetic_tier_at_most_2(self):
        rec = _sample(
            source_dataset="SlideVQA",
            query="What is the slide title?",
        )
        prior = _compute_prior(rec)
        assert prior["tier_prior"] <= 2

    # ── CORD / SROIE ─────────────────────────────────────────────────────────────

    def test_cord_tier_at_most_2(self):
        rec = _sample(
            source_dataset="CORD",
            query="Cross-reference figures across pages to calculate the total.",
            has_chart=True,
            has_figure=True,
            num_pages=4,
        )
        prior = _compute_prior(rec)
        assert prior["tier_prior"] <= 2

    def test_sroie_tier_at_most_2(self):
        rec = _sample(
            source_dataset="SROIE",
            query="Cross-reference figures across pages to calculate the total.",
            has_chart=True,
            has_figure=True,
            num_pages=4,
        )
        prior = _compute_prior(rec)
        assert prior["tier_prior"] <= 2

    # ── TabFact ──────────────────────────────────────────────────────────────────

    def test_tabfact_rds_at_least_2(self):
        # Even for a short, simple query, TabFact requires rds >= 2
        rec = _sample(source_dataset="TabFact", query="Is it true?")
        prior = _compute_prior(rec)
        assert prior["rds_est"] >= 2

    # ── WikiTableQuestions ───────────────────────────────────────────────────────

    def test_wtq_rds_at_least_3(self):
        # WTQ always gets rds >= 3
        rec = _sample(source_dataset="WikiTableQuestions", query="What is the name?")
        prior = _compute_prior(rec)
        assert prior["rds_est"] >= 3

    def test_wtq_alias_wtq_also_gets_rds_3(self):
        # "WTQ" shorthand
        rec = _sample(source_dataset="WTQ", query="What is the name?")
        prior = _compute_prior(rec)
        assert prior["rds_est"] >= 3


# ===========================================================================
# Tier boundary consistency
# ===========================================================================

class TestTierBoundaries:
    """Composite score → tier mapping follows _TIER1_MAX and _TIER2_MAX."""

    def test_composite_below_tier1_max_gives_tier1(self):
        # VDS=1, RDS=1, SES=1 → composite = 1.0 < TIER1_MAX (2.2)
        c = _compute_composite(1, 1, 1)
        assert c < _TIER1_MAX
        assert _composite_to_tier(c) == 1

    def test_composite_above_tier2_max_gives_tier3(self):
        # VDS=4, RDS=4, SES=4 → composite = 4.0 >= TIER2_MAX (3.4)
        c = _compute_composite(4, 4, 4)
        assert c >= _TIER2_MAX
        assert _composite_to_tier(c) == 3

    def test_composite_between_boundaries_gives_tier2(self):
        # Exactly TIER1_MAX = 2.2 should map to Tier 2
        assert _composite_to_tier(_TIER1_MAX) == 2
        # Just below TIER2_MAX should also be Tier 2
        assert _composite_to_tier(_TIER2_MAX - 0.01) == 2

    def test_dataset_override_can_change_tier_from_composite(self):
        # RVL-CDIP forces tier=1 even if composite would suggest Tier 3
        rec = _sample(
            source_dataset="RVL-CDIP",
            has_chart=True,
            has_figure=True,
            num_pages=5,
            query="Cross-reference figures across pages to calculate total.",
        )
        prior = _compute_prior(rec)
        # composite is high, but tier_prior must still be 1
        assert prior["tier_prior"] == 1
        # composite itself is recomputed from overridden scores, not capped
        assert isinstance(prior["composite_est"], float)


# ===========================================================================
# Idempotency / purity
# ===========================================================================

class TestPurity:
    """_compute_prior must be pure and not mutate its input."""

    def test_does_not_mutate_input_record(self):
        rec = _sample()
        original = dict(rec)
        _compute_prior(rec)
        assert rec == original

    def test_same_input_same_output(self):
        rec = _sample(
            query="Calculate the total from the table.",
            has_table=True,
            num_pages=2,
        )
        prior1 = _compute_prior(rec)
        prior2 = _compute_prior(rec)
        assert prior1 == prior2

    def test_different_queries_can_differ(self):
        easy = _sample(query="What is the title?")
        hard = _sample(
            query="Cross-reference figures across pages to calculate total.",
            has_chart=True,
            has_figure=True,
            num_pages=3,
        )
        p_easy = _compute_prior(easy)
        p_hard = _compute_prior(hard)
        # Hard sample should have strictly higher composite
        assert p_hard["composite_est"] > p_easy["composite_est"]
