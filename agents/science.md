# Agent 5 -- Science / Technology

**Agent id**: `science`
**Isolation policy**: `orchestrator/isolation.py::POLICIES["science"]`
**Implementation**: `src/investment_research/agents/science.py`

## Purpose

Assess the asset on checkable attributes, not on the strength of the mechanism story.

## MUST see

- The shared verified evidence set.

## MUST NOT see

- Every agent's evaluation.

## Output contract

Biotech: mechanism, target validation, trial design, sample size, randomization, blinding, control, primary and secondary endpoints, statistical power, effect size, safety, subgroup and post-hoc framing, competitor data, standard of care.

Technology: product architecture, patents, switching cost, deployment evidence, customer validation, technical differentiation, replicability, integration, TRL, production scalability.

## Hard rules

1. State explicitly whether the primary endpoint is a **clinical outcome** or a **surrogate/biomarker**. This distinction decides more theses than the mechanism does.
2. A surrogate endpoint only supports approval if the regulator accepts it as reasonably likely to predict benefit. That acceptance is a separate, independently verifiable fact -- never assume it.
3. Independent literature that contradicts the surrogate is a critical finding, not a footnote.
4. For technology: differentiation that never appears in deployments, customers or patents is an assertion, not a moat.
