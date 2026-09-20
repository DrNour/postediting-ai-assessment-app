"""Schema-aware safeguards for EduApp statistical analyses.

The validator is deliberately separate from SciPy/Statsmodels execution.  It
inspects an analysis request and returns an auditable ``allow``, ``warn``, or
``block`` decision before an inferential test is run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import re
from typing import Any, Iterable

import numpy as np
import pandas as pd


POLICY_VERSION = "0.1.0"
MIN_INFERENTIAL_N = 5
RECOMMENDED_N = 10

_DECISION_RANK = {"allow": 0, "warn": 1, "block": 2}
_PARTICIPANT_COLUMNS = (
    "student_id",
    "participant_id",
    "learner_id",
    "user_id",
    "student_name",
)
_KNOWN_IDENTIFIERS = {
    "id",
    "uuid",
    "submission_id",
    "assignment_id",
    "student_id",
    "participant_id",
    "learner_id",
    "user_id",
    "record_id",
    "row_id",
}
_UUID_PATTERN = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class GuardFinding:
    """One rule outcome returned by EduStatGuard."""

    rule_id: str
    decision: str
    title: str
    explanation: str
    repair: str
    evidence: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "decision": self.decision,
            "title": self.title,
            "explanation": self.explanation,
            "repair": self.repair,
            "evidence": self.evidence,
        }


@dataclass(frozen=True)
class GuardReport:
    """Complete validation result for one requested analysis."""

    analysis: str
    findings: tuple[GuardFinding, ...] = ()
    policy_version: str = POLICY_VERSION

    @property
    def decision(self) -> str:
        if not self.findings:
            return "allow"
        return max(
            (finding.decision for finding in self.findings),
            key=lambda value: _DECISION_RANK[value],
        )

    @property
    def blocked(self) -> bool:
        return self.decision == "block"

    def as_records(self) -> list[dict[str, Any]]:
        return [finding.as_dict() for finding in self.findings]


def _normalise_name(column: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(column).strip().lower()).strip("_")


def _nonempty(series: pd.Series) -> pd.Series:
    values = series.dropna().astype(str).str.strip()
    return values[~values.str.lower().isin({"", "none", "nan", "null"})]


def infer_variable_role(column: str, series: pd.Series) -> str:
    """Infer a conservative semantic role from a name and observed values."""

    name = _normalise_name(column)
    values = _nonempty(series)

    if name in _KNOWN_IDENTIFIERS or name.endswith("_uuid"):
        return "identifier"

    if not values.empty:
        uuid_share = float(values.str.match(_UUID_PATTERN).mean())
        uniqueness = float(values.nunique(dropna=True) / len(values))

        if uuid_share >= 0.8:
            return "identifier"

        # Names ending in _id or _code are treated as keys only when the
        # observed values are sufficiently high-cardinality.  This avoids
        # misclassifying small experimental condition codes.
        if (name.endswith("_id") or name.endswith("_code")) and uniqueness >= 0.5:
            return "identifier"

    numeric = pd.to_numeric(series, errors="coerce")
    if int(numeric.notna().sum()) >= max(3, int(math.ceil(len(series) * 0.5))):
        return "numeric"

    unique_count = values.nunique(dropna=True)
    if 2 <= unique_count <= 50:
        return "categorical"
    return "text"


def find_participant_column(df: pd.DataFrame) -> str | None:
    """Return the first recognised participant key present in the data."""

    normalised = {_normalise_name(column): column for column in df.columns}
    for candidate in _PARTICIPANT_COLUMNS:
        if candidate in normalised:
            return normalised[candidate]
    return None


def _effective_n_finding(n_complete: int, n_total: int) -> GuardFinding | None:
    evidence = {
        "complete_cases": int(n_complete),
        "selected_rows": int(n_total),
        "excluded_rows": int(max(0, n_total - n_complete)),
    }
    if n_complete < MIN_INFERENTIAL_N:
        return GuardFinding(
            rule_id="ESG-N-001",
            decision="block",
            title="Too few complete observations",
            explanation=(
                f"Only {n_complete} complete observations are available. "
                "An inferential result at this size is too unstable to present as evidence."
            ),
            repair="Treat the plot and values as descriptive, or collect more complete observations.",
            evidence=evidence,
        )
    if n_complete < RECOMMENDED_N:
        return GuardFinding(
            rule_id="ESG-N-002",
            decision="warn",
            title="Very small effective sample",
            explanation=(
                f"The test uses {n_complete} complete observations. The estimate and p-value "
                "may change substantially with one additional case."
            ),
            repair="Report the exact n and effect size, use cautious language, and add data if possible.",
            evidence=evidence,
        )
    return None


def _repeated_participant_finding(
    df: pd.DataFrame,
    complete_mask: pd.Series,
) -> GuardFinding | None:
    participant_col = find_participant_column(df)
    if participant_col is None:
        return None

    participant_values = _nonempty(df.loc[complete_mask, participant_col])
    if participant_values.empty:
        return None

    observations = int(len(participant_values))
    participants = int(participant_values.nunique(dropna=True))
    repeated = observations - participants
    if repeated <= 0:
        return None

    return GuardFinding(
        rule_id="ESG-REP-001",
        decision="warn",
        title="Repeated observations detected",
        explanation=(
            f"The {observations} complete rows represent {participants} unique participants. "
            "Ordinary tests treat rows as independent and may understate uncertainty."
        ),
        repair=(
            f"Aggregate to one row per {participant_col}, or use a repeated-measures/mixed-effects "
            "analysis with the participant key as the cluster."
        ),
        evidence={
            "participant_column": participant_col,
            "complete_rows_with_participant": observations,
            "unique_participants": participants,
            "repeated_rows": repeated,
        },
    )


def _identifier_finding(column: str, role: str) -> GuardFinding | None:
    if role != "identifier":
        return None
    return GuardFinding(
        rule_id="ESG-ROLE-001",
        decision="block",
        title="Identifier selected as an analytical variable",
        explanation=(
            f"'{column}' appears to be a record or entity identifier. Its codes do not measure "
            "a construct, so an inferential association would be structurally misleading."
        ),
        repair="Choose a substantive outcome, treatment, group, or ordered variable instead.",
        evidence={"column": column, "inferred_role": role},
    )


def validate_numeric_pair(
    df: pd.DataFrame,
    x_column: str,
    y_column: str,
    analysis: str,
) -> GuardReport:
    """Validate a paired test or correlation request before execution."""

    findings: list[GuardFinding] = []
    missing = [column for column in (x_column, y_column) if column not in df.columns]
    if missing:
        findings.append(
            GuardFinding(
                rule_id="ESG-SCHEMA-001",
                decision="block",
                title="Requested column is missing",
                explanation=f"The dataset does not contain: {', '.join(missing)}.",
                repair="Refresh the schema or select columns that exist in the filtered dataset.",
                evidence={"missing_columns": missing},
            )
        )
        return GuardReport(analysis=analysis, findings=tuple(findings))

    if x_column == y_column:
        findings.append(
            GuardFinding(
                rule_id="ESG-PAIR-001",
                decision="block",
                title="The same variable was selected twice",
                explanation="A variable compared or correlated with itself cannot answer the requested question.",
                repair="Select two distinct variables.",
                evidence={"column": x_column},
            )
        )

    for column in dict.fromkeys((x_column, y_column)):
        role = infer_variable_role(column, df[column])
        finding = _identifier_finding(column, role)
        if finding is not None:
            findings.append(finding)

    numeric = df[[x_column, y_column]].apply(pd.to_numeric, errors="coerce")
    complete_mask = numeric.notna().all(axis=1)
    complete = numeric.loc[complete_mask]
    n_finding = _effective_n_finding(len(complete), len(df))
    if n_finding is not None:
        findings.append(n_finding)

    for column in (x_column, y_column):
        if len(complete) and complete[column].nunique(dropna=True) < 2:
            findings.append(
                GuardFinding(
                    rule_id="ESG-SCALE-001",
                    decision="block",
                    title="No variation in a selected variable",
                    explanation=f"'{column}' is constant across the complete observations.",
                    repair="Choose a variable with at least two observed values.",
                    evidence={"column": column, "unique_values": int(complete[column].nunique())},
                )
            )

    repeated = _repeated_participant_finding(df, complete_mask)
    if repeated is not None:
        findings.append(repeated)

    return GuardReport(analysis=analysis, findings=tuple(findings))


def _functional_dependency(frame: pd.DataFrame, source: str, target: str) -> bool:
    if frame.empty or frame[source].nunique() < 2 or frame[target].nunique() < 2:
        return False
    return bool(frame.groupby(source, dropna=False)[target].nunique(dropna=False).max() == 1)


def validate_contingency(
    df: pd.DataFrame,
    row_column: str,
    column_column: str,
) -> GuardReport:
    """Validate a categorical association request and its expected counts."""

    findings: list[GuardFinding] = []
    missing = [column for column in (row_column, column_column) if column not in df.columns]
    if missing:
        findings.append(
            GuardFinding(
                rule_id="ESG-SCHEMA-001",
                decision="block",
                title="Requested column is missing",
                explanation=f"The dataset does not contain: {', '.join(missing)}.",
                repair="Refresh the schema or select columns that exist in the filtered dataset.",
                evidence={"missing_columns": missing},
            )
        )
        return GuardReport(analysis="categorical_association", findings=tuple(findings))

    if row_column == column_column:
        findings.append(
            GuardFinding(
                rule_id="ESG-CAT-001",
                decision="block",
                title="The same categorical variable was selected twice",
                explanation="A cross-tabulation of a variable with itself is deterministic by construction.",
                repair="Select two distinct categorical variables.",
                evidence={"column": row_column},
            )
        )

    roles = {
        row_column: infer_variable_role(row_column, df[row_column]),
        column_column: infer_variable_role(column_column, df[column_column]),
    }
    for column, role in roles.items():
        finding = _identifier_finding(column, role)
        if finding is not None:
            findings.append(finding)

    frame = df[[row_column, column_column]].dropna().copy()
    for column in (row_column, column_column):
        frame[column] = frame[column].astype(str).str.strip()
        frame = frame[~frame[column].str.lower().isin({"", "none", "nan", "null"})]

    complete_mask = df.index.isin(frame.index)
    complete_mask = pd.Series(complete_mask, index=df.index)
    n_finding = _effective_n_finding(len(frame), len(df))
    if n_finding is not None:
        findings.append(n_finding)

    table = pd.crosstab(frame[row_column], frame[column_column])
    if table.shape[0] < 2 or table.shape[1] < 2:
        findings.append(
            GuardFinding(
                rule_id="ESG-CAT-002",
                decision="block",
                title="Insufficient category variation",
                explanation="Both variables need at least two observed categories.",
                repair="Change the filters or choose variables with two or more populated levels.",
                evidence={"table_rows": int(table.shape[0]), "table_columns": int(table.shape[1])},
            )
        )
    elif len(frame):
        a_to_b = _functional_dependency(frame, row_column, column_column)
        b_to_a = _functional_dependency(frame, column_column, row_column)
        if a_to_b or b_to_a:
            determinant = row_column if a_to_b else column_column
            dependent = column_column if a_to_b else row_column
            structural = roles.get(determinant) == "identifier" or roles.get(dependent) == "identifier"
            findings.append(
                GuardFinding(
                    rule_id="ESG-FD-001",
                    decision="block" if structural else "warn",
                    title="Deterministic mapping detected",
                    explanation=(
                        f"Each observed '{determinant}' maps to only one '{dependent}'. "
                        "The apparent association may come from the data schema or task design rather than evidence of a stochastic relationship."
                    ),
                    repair="Use the table descriptively, or test variables that are not defined or nested by one another.",
                    evidence={
                        "determinant": determinant,
                        "dependent": dependent,
                        "structural_identifier_present": structural,
                        "row_determines_column": a_to_b,
                        "column_determines_row": b_to_a,
                    },
                )
            )

        observed = table.to_numpy(dtype=float)
        total = float(observed.sum())
        expected = np.outer(observed.sum(axis=1), observed.sum(axis=0)) / total
        below_five = int((expected < 5).sum())
        cells = int(expected.size)
        below_five_share = float(below_five / cells)
        minimum_expected = float(expected.min())

        if minimum_expected < 1 or below_five_share > 0.5:
            is_two_by_two = table.shape == (2, 2)
            findings.append(
                GuardFinding(
                    rule_id="ESG-SPARSE-001",
                    decision="warn" if is_two_by_two else "block",
                    title="Contingency table is too sparse for Pearson's chi-square",
                    explanation=(
                        f"{below_five} of {cells} expected cells ({below_five_share:.0%}) are below 5; "
                        f"the minimum expected count is {minimum_expected:.3g}."
                    ),
                    repair=(
                        "Use Fisher's exact test for this 2x2 table."
                        if is_two_by_two
                        else "Combine defensible categories, collect more observations, or use a suitable exact/model-based method."
                    ),
                    evidence={
                        "table_shape": [int(table.shape[0]), int(table.shape[1])],
                        "cells_below_5": below_five,
                        "share_below_5": below_five_share,
                        "minimum_expected_count": minimum_expected,
                    },
                )
            )
        elif below_five_share > 0.2:
            findings.append(
                GuardFinding(
                    rule_id="ESG-SPARSE-002",
                    decision="warn",
                    title="Some expected cell counts are small",
                    explanation=(
                        f"{below_five} of {cells} expected cells ({below_five_share:.0%}) are below 5. "
                        "The chi-square approximation may be inaccurate."
                    ),
                    repair="Consider combining defensible categories or increasing the sample.",
                    evidence={
                        "table_shape": [int(table.shape[0]), int(table.shape[1])],
                        "cells_below_5": below_five,
                        "share_below_5": below_five_share,
                        "minimum_expected_count": minimum_expected,
                    },
                )
            )

    repeated = _repeated_participant_finding(df, complete_mask)
    if repeated is not None:
        findings.append(repeated)

    return GuardReport(analysis="categorical_association", findings=tuple(findings))


def format_p_value(value: Any) -> str:
    """Return an APA-style p-value without ever displaying a valid p as zero."""

    try:
        p_value = float(value)
    except (TypeError, ValueError):
        return "NA"

    if not math.isfinite(p_value) or p_value < 0 or p_value > 1:
        return "NA"
    if p_value < 0.001:
        return "< .001"

    rendered = f"{p_value:.3f}"
    if rendered.startswith("0."):
        rendered = rendered[1:]
    return f"= {rendered}"


def format_p_value_columns(
    frame: pd.DataFrame,
    columns: Iterable[str] = ("p_value", "raw_p_value", "adjusted_p_value"),
) -> pd.DataFrame:
    """Copy a results table and format every available p-value column."""

    formatted = frame.copy()
    for column in columns:
        if column in formatted.columns:
            formatted[column] = formatted[column].map(format_p_value)
    return formatted

