# External Source Inventory

## Usage Labels

- `A`: can be used directly as a Level A source
- `B`: can be used directly as a Level B source
- `raw_material`: useful only after re-annotation, filtering, or task reframing

## Inventory

| Source | Primary Level | Usage | Status | License | Notes |
| --- | --- | --- | --- | --- | --- |
| `chemu_2020` | `A` | `raw_material` | `downloaded` | `CC BY-NC 3.0` | Patent IE corpus for reagent roles and reaction description; needs benchmark reframing. |
| `choriso` | `A` | `raw_material` | `downloaded` | `CC BY 4.0` | Reaction prediction and selectivity source; better as re-annotation material than direct benchmark. |
| `pmechdb` | `A` | `A` | `downloaded` | `CC BY-NC-ND 4.0` | Direct mechanistic source for Level A. |
| `rmechdb` | `A` | `A` | `blocked` | `CC BY-NC-ND 4.0` | Direct Level A source, but official access is request form plus email delivery. |
| `uspto_50k` | `A` | `A` | `downloaded` | `CC BY 4.0` | Direct reaction classification source for Level A. |
| `uspto_llm` | `A` | `raw_material` | `downloaded` | `CC BY 4.0` | Information-enriched reaction dataset; LLM-assisted labeling makes it better as raw material. |
| `weave2` | `A` | `raw_material` | `downloaded` | `Unknown` | Procedural patent IE corpus, not a direct chemistry benchmark. |
| `lowe_uspto` | `B` | `raw_material` | `downloaded` | `CC0` | Core raw reaction corpus for custom Level B curation. |
| `orderly` | `B` | `B` | `downloaded` | `CC BY 4.0` | Direct open one-step retrosynthesis benchmark. |
| `paroutes` | `B` | `B` | `downloaded` | `Apache-2.0` | Direct Level B ecosystem source; current local artifact is the repository snapshot. |
| `ord` | `shared` | `raw_material` | `downloaded` | `CC BY-SA 4.0 data; Apache-2.0 code` | Structured reaction record store for both A and B after filtering. |
| `pistachio` | `none` | `raw_material` | `blocked` | `Commercial` | Commercial source, not downloaded. |
| `reaxys` | `none` | `raw_material` | `blocked` | `Commercial` | Commercial source, not downloaded. |

## Size Summary

| Source | Raw Files | Raw Size Bytes | Extracted Files | Extracted Size Bytes |
| --- | --- | --- | --- | --- |
| `chemu_2020` | 9 | 4110914 | 222 | 202006 |
| `choriso` | 3 | 630692320 | 5 | 3506113509 |
| `pmechdb` | 2 | 7752148 | 5 | 72126253 |
| `rmechdb` | 0 | 0 | 0 | 0 |
| `uspto_50k` | 3 | 20542486 | 0 | 0 |
| `uspto_llm` | 3 | 702148927 | 14 | 15340765768 |
| `weave2` | 1 | 1261146 | 66 | 5553703 |
| `lowe_uspto` | 3 | 162790816 | 4 | 1695809051 |
| `orderly` | 4 | 576739002 | 16 | 3778323 |
| `paroutes` | 1 | 1591829 | 38 | 3307160 |
| `ord` | 1 | 1175492752 | 559 | 1176044355 |

Detailed machine-readable metadata is in `inventory.csv`.
