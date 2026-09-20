# EduStatGuard policy v0.1.0

EduStatGuard validates a statistical request before the analysis engine runs. It returns one of three decisions:

- `allow`: no current rule prevents the analysis;
- `warn`: the analysis may run, but the limitation and repair advice must be shown;
- `block`: inferential output is suppressed, while descriptive summaries may still be shown.

## Initial rule catalogue

| Rule | Decision | Trigger | Repair |
|---|---|---|---|
| ESG-ROLE-001 | Block | An identifier, UUID, or high-cardinality key is selected as an analytical variable | Select a substantive outcome, treatment, or group variable |
| ESG-FD-001 | Warn or block | One selected category deterministically maps to the other; block when an identifier is involved | Use the table descriptively or select variables not defined by one another |
| ESG-REP-001 | Warn | Complete rows contain repeated participant keys | Aggregate per participant or use a repeated-measures/mixed-effects model |
| ESG-SPARSE-001/002 | Warn or block | Expected contingency counts violate chi-square approximation rules | Use Fisher's exact test for 2x2 data, combine defensible levels, collect data, or use another model |
| ESG-N-001/002 | Block below 5; warn below 10 | Too few complete cases remain after pairwise deletion | Treat as descriptive or collect more complete observations |
| p-value display | Formatting safeguard | A p-value is smaller than .001, including numerical underflow to zero | Display `p < .001`; never display `p = 0` |

The validator is separated from statistical execution so the policy can be versioned, tested, and evaluated independently for the EduStatGuard systems study.
