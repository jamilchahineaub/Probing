# Literature ledger

## Status

No peer-reviewed paper was used to derive or tune EXP-0001; its equations came
from the project research plan and the standard Kelvin–Voigt constitutive
identity stated there.

`InitialPlan.md` names relevant authors and claims but its citations are encoded
as internal tokens such as `turn16search0`, not DOI/arXiv/URL records. Those
claims are treated as research-plan guidance, not as independently verified
literature evidence. They must be resolved to primary sources before they are
used in a baseline implementation, comparison, or public claim.

## Primary-source audit queue from the research plan

| Work named in plan | Claimed relevance in plan | DOI/arXiv/URL | Audit status | Reproduction/distinction need |
|---|---|---|---|---|
| Bodie et al. | Aerial inspection with impedance/direct force control | Unresolved | Not audited | Establish known/rigid force-control baseline |
| Tzoumanikas et al. | Hybrid force/position NMPC for aerial manipulation | Unresolved | Not audited | Establish interaction-aware NMPC baseline |
| Brunner et al. | Passivity/energy-tank interaction with poorly modelled moving environments | Unresolved | Not audited | Compare safety/feasibility benefit of explicit identification |
| Zhang et al. | Learned variable impedance on uncertain heterogeneous surfaces | Unresolved | Not audited | Avoid unsupported adaptive-impedance novelty claim |
| Aucone et al. | Aerial traversal of unknown compliant objects/branches | Unresolved | Not audited | Closest unknown-compliance conceptual comparison |
| Sathyanarayan and Abraham | Contact-aware Fisher-information probing | Unresolved | Not audited | Distinguish aerial, bounded, decision-oriented probe design |
| Naser et al. | Sensorless aerial wrench observation | Unresolved | Not audited | Later sensorless observer comparison if that extension is pursued |
| Brummelhuis et al. | Proprioceptive UAV contact inference | Unresolved | Not audited | Delimit any later sensorless/contact-localization claim |

No reported result, assumption, controller detail, environment, or limitation
from these entries should be copied from the plan into scientific prose until
the corresponding primary source has been read and added here with a stable
identifier.

