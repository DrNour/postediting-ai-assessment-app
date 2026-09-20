import unittest

import numpy as np
import pandas as pd

from modules.edustatguard import (
    format_p_value,
    format_p_value_columns,
    infer_variable_role,
    validate_contingency,
    validate_numeric_pair,
)
from modules.rtl_layout import is_arabic_dominant


class VariableRoleTests(unittest.TestCase):
    def test_named_identifier_is_blocked(self):
        values = pd.Series(["a", "b", "c"])
        self.assertEqual(infer_variable_role("assignment_id", values), "identifier")

    def test_uuid_values_are_identifiers(self):
        values = pd.Series(
            [
                "028cab9a-75df-44b0-93cb-4155c138ef06",
                "08fc7d30-3e3c-4500-b79d-63d7f28c4364",
                "13f588ac-02c2-4b64-b0da-5945f9b1b56d",
            ]
        )
        self.assertEqual(infer_variable_role("record", values), "identifier")

    def test_metric_is_numeric(self):
        values = pd.Series([0.2, 0.4, 0.8, 0.9])
        self.assertEqual(infer_variable_role("bleu", values), "numeric")


class NumericPairValidationTests(unittest.TestCase):
    def test_three_case_correlation_is_blocked(self):
        df = pd.DataFrame({"x": [1, 2, 3], "y": [3, 2, 1]})
        report = validate_numeric_pair(df, "x", "y", "correlation")
        self.assertEqual(report.decision, "block")
        self.assertIn("ESG-N-001", {item.rule_id for item in report.findings})

    def test_seven_case_paired_test_warns(self):
        df = pd.DataFrame({"x": range(7), "y": range(1, 8)})
        report = validate_numeric_pair(df, "x", "y", "paired")
        self.assertEqual(report.decision, "warn")
        self.assertIn("ESG-N-002", {item.rule_id for item in report.findings})

    def test_repeated_students_are_detected(self):
        df = pd.DataFrame(
            {
                "student_id": ["s1", "s1", "s2", "s2", "s3", "s3", "s4", "s4", "s5", "s5"],
                "x": np.arange(10),
                "y": np.arange(10) + 0.5,
            }
        )
        report = validate_numeric_pair(df, "x", "y", "correlation")
        self.assertEqual(report.decision, "warn")
        self.assertIn("ESG-REP-001", {item.rule_id for item in report.findings})

    def test_identifier_cannot_be_numeric_outcome(self):
        df = pd.DataFrame({"student_id": range(10), "score": range(10, 20)})
        report = validate_numeric_pair(df, "student_id", "score", "correlation")
        self.assertEqual(report.decision, "block")
        self.assertIn("ESG-ROLE-001", {item.rule_id for item in report.findings})


class ContingencyValidationTests(unittest.TestCase):
    def test_assignment_id_by_topic_is_blocked(self):
        df = pd.DataFrame(
            {
                "assignment_id": [f"assignment-{i}" for i in range(12) for _ in range(3)],
                "topic": [topic for topic in ("AI", "Legal", "Health") for _ in range(12)],
            }
        )
        report = validate_contingency(df, "assignment_id", "topic")
        self.assertEqual(report.decision, "block")
        rule_ids = {item.rule_id for item in report.findings}
        self.assertIn("ESG-ROLE-001", rule_ids)
        self.assertIn("ESG-FD-001", rule_ids)

    def test_balanced_table_is_allowed(self):
        rows = []
        for group in ("A", "B"):
            for outcome in ("yes", "no"):
                rows.extend({"group": group, "outcome": outcome} for _ in range(10))
        report = validate_contingency(pd.DataFrame(rows), "group", "outcome")
        self.assertEqual(report.decision, "allow")

    def test_large_sparse_table_is_blocked(self):
        df = pd.DataFrame(
            {
                "group": [f"g{i}" for i in range(12)],
                "outcome": ["yes", "no"] * 6,
            }
        )
        report = validate_contingency(df, "group", "outcome")
        self.assertEqual(report.decision, "block")
        self.assertIn("ESG-SPARSE-001", {item.rule_id for item in report.findings})


class PValueFormattingTests(unittest.TestCase):
    def test_zero_is_never_printed_as_zero(self):
        self.assertEqual(format_p_value(0), "< .001")
        self.assertEqual(format_p_value(0.0005), "< .001")

    def test_regular_p_value_is_rounded(self):
        self.assertEqual(format_p_value(0.0156), "= .016")

    def test_dataframe_columns_are_formatted(self):
        frame = pd.DataFrame({"p_value": [0.0, 0.0156], "n": [3, 7]})
        result = format_p_value_columns(frame)
        self.assertEqual(result["p_value"].tolist(), ["< .001", "= .016"])
        self.assertEqual(result["n"].tolist(), [3, 7])


class ArabicLayoutTests(unittest.TestCase):
    def test_arabic_text_is_detected(self):
        self.assertTrue(is_arabic_dominant("زار الوزير الجامعة أمس."))

    def test_english_text_remains_ltr(self):
        self.assertFalse(is_arabic_dominant("The minister visited the university."))

    def test_arabic_dominant_mixed_text_is_detected(self):
        self.assertTrue(is_arabic_dominant("استخدم مصطلح AI في هذه الجملة العربية الطويلة."))


if __name__ == "__main__":
    unittest.main()
