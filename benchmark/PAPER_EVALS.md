# Paper Eval Subsets

These files are compact paper-oriented evaluation subsets built on top of the large benchmark pools.

## Files

- `level_a_eval.jsonl`
- `level_b_eval.jsonl`
- `level_c_eval.jsonl`
- `paper_eval_manifest.yaml`

## Intended Sizes

- `Level A`: 420 records
- `Level B`: 420 records
- `Level C`: 150 records

## Balancing Policy

### Level A

Balanced by source and task subtype:

- `PMechDB` reaction center identification
- `PMechDB` mechanistic classification
- `USPTO-50K` transformation classification
- `ChEMU 2020` reagent-role extraction
- `WEAVE2` reagent-role extraction
- `CHORISO` condition-role identification

### Level B

Balanced by source and difficulty:

- `ORDerly`: easy / medium / hard
- `PaRoutes selected_reactions_all`: easy / medium / hard

The paper-facing judge treats source precursor sets as documented reference
answers. Level B scores plausible immediate one-step retrosynthesis rather than
exact recovery of the sampled source route.

### Level C

Balanced by route source and difficulty:

- all internal pilot route-design tasks are kept
- `PaRoutes n1`: medium / hard
- `PaRoutes n5`: medium / hard

## Determinism

Sampling is deterministic and based on a stable hash of `record_id`.
