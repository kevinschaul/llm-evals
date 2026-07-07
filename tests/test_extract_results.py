"""Unit tests for the pure helpers in extract_results.py."""
from types import SimpleNamespace

import pytest

from extract_results import (
    derive_passed,
    extract_checks,
    extract_tokens,
    json_safe,
    normalize_text,
    primary_scorer_name,
    score_to_passed,
    task_matches_eval,
)


def score(value=None, explanation=None, metadata=None):
    return SimpleNamespace(value=value, explanation=explanation, metadata=metadata)


class TestTaskMatchesEval:
    def test_underscores_match_hyphens(self):
        assert task_matches_eval("article_tracking_trump", "article-tracking-trump")

    def test_substring_does_not_match(self):
        assert not task_matches_eval("article_tracking_trump", "article-tracking")
        assert not task_matches_eval("grab_bag_2", "grab-bag")


class TestScoreToPassed:
    @pytest.mark.parametrize(
        "value,expected",
        [
            ("C", True),
            ("I", False),
            (1.0, True),
            (1, True),
            (0.8, False),
            (0.0, False),
            (True, True),
            (False, False),
            (None, None),
            ({"weird": "value"}, False),
        ],
    )
    def test_values(self, value, expected):
        assert score_to_passed(value) is expected


class TestDerivePassed:
    def test_unscored(self):
        assert derive_passed({}, None) is None

    def test_uses_primary_scorer(self):
        scores = {"includes": score("I"), "llm_judge": score("C")}
        assert derive_passed(scores, "llm_judge") is True
        assert derive_passed(scores, "includes") is False

    def test_falls_back_past_git_diff(self):
        scores = {"git_diff": score("C"), "check_output": score(0.0)}
        assert derive_passed(scores, None) is False

    def test_git_diff_only_is_unscored(self):
        assert derive_passed({"git_diff": score("C")}, None) is None


class TestExtractChecks:
    def test_prefers_metadata(self):
        scores = {
            "check_output": score(
                0.5,
                explanation="✓ from_text_should_be_ignored",
                metadata={"checks": {"has_py_file": True, "row_count": False}},
            )
        }
        assert extract_checks(scores) == [
            {"name": "has_py_file", "passed": True},
            {"name": "row_count", "passed": False},
        ]

    def test_parses_explanation_lines(self):
        explanation = "✓ has_py_file\n✗ row_count_correct\n  (got 3, expected 5)"
        scores = {"check_output": score(0.5, explanation=explanation)}
        assert extract_checks(scores) == [
            {"name": "has_py_file", "passed": True},
            {"name": "row_count_correct", "passed": False},
        ]

    def test_ignores_git_diff(self):
        scores = {"git_diff": score("C", explanation="✓ looks like a check")}
        assert extract_checks(scores) == []

    def test_no_checks(self):
        scores = {"includes": score("C", explanation="matched expected output")}
        assert extract_checks(scores) == []


class TestPrimaryScorerName:
    def _log(self, names):
        return SimpleNamespace(
            results=SimpleNamespace(
                scores=[SimpleNamespace(name=n) for n in names]
            )
        )

    def test_skips_git_diff(self):
        assert primary_scorer_name(self._log(["git_diff", "check_output"])) == "check_output"

    def test_declared_order_wins(self):
        assert primary_scorer_name(self._log(["includes", "llm_judge"])) == "includes"

    def test_none_when_diff_only(self):
        assert primary_scorer_name(self._log(["git_diff"])) is None

    def test_none_without_results(self):
        assert primary_scorer_name(SimpleNamespace(results=None)) is None


class TestExtractTokens:
    def _usage(self, inp, out):
        return SimpleNamespace(input_tokens=inp, output_tokens=out)

    def test_prefers_model_usage(self):
        sample = SimpleNamespace(
            model_usage={"anthropic/claude": self._usage(500, 100)},
            output=SimpleNamespace(usage=self._usage(1, 1)),
        )
        assert extract_tokens(sample) == (500, 100)

    def test_sums_multiple_models(self):
        sample = SimpleNamespace(
            model_usage={"a": self._usage(500, 100), "b": self._usage(200, 50)},
            output=None,
        )
        assert extract_tokens(sample) == (700, 150)

    def test_falls_back_to_output_usage(self):
        sample = SimpleNamespace(
            model_usage={},
            output=SimpleNamespace(usage=self._usage(300, 40)),
        )
        assert extract_tokens(sample) == (300, 40)

    def test_zero_usage_is_missing(self):
        sample = SimpleNamespace(
            model_usage={"a": self._usage(0, 0)},
            output=SimpleNamespace(usage=self._usage(0, 0)),
        )
        assert extract_tokens(sample) == (None, None)

    def test_no_usage_recorded(self):
        sample = SimpleNamespace(model_usage=None, output=None)
        assert extract_tokens(sample) == (None, None)


class TestNormalizeText:
    def test_string_passthrough(self):
        assert normalize_text("hello") == "hello"

    def test_none(self):
        assert normalize_text(None) == ""

    def test_chat_messages(self):
        messages = [
            SimpleNamespace(content="plain text"),
            SimpleNamespace(
                content=[
                    SimpleNamespace(text="part one"),
                    SimpleNamespace(type="image"),
                ]
            ),
        ]
        assert normalize_text(messages) == "plain text\npart one\n[image]"


class TestJsonSafe:
    def test_passthrough(self):
        assert json_safe({"a": [1, "x", None]}) == {"a": [1, "x", None]}

    def test_coerces_objects(self):
        assert json_safe(SimpleNamespace(a=1)) == "namespace(a=1)"
