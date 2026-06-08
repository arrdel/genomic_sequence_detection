We thank Reviewer wkYq for the careful reading and constructive tone.

**On venue fit and clinical relevance.** MLHC's Novel Methods theme explicitly covers unsupervised representation learning for healthcare data. Wastewater epidemiology informed population-level public health decisions throughout COVID-19, with surveillance programs now operational across the US, EU, and internationally. The revised title and abstract reframe our contribution as anomaly detection infrastructure that flags samples for targeted clinical sequencing — not a clinical decision tool itself.

**On the anomaly detection experimental design.** We ran this experiment in full. Omicron reads were injected at controlled rates into a Wuhan baseline stream with explicit positive labels; anomaly score = L2 distance to a held-out Wuhan centroid.

| Injection rate | n injected | AUROC | AUPRC | TPR @ 5% FPR |
|----------------|------------|-------|-------|--------------|
| 1%  |    51 | 0.522 | 0.010 | 0.000 |
| 5%  |   263 | 0.498 | 0.050 | 0.053 |
| 10% |   556 | 0.515 | 0.104 | 0.058 |
| 20% | 1,250 | 0.508 | 0.209 | 0.060 |
| 50% | 5,000 | 0.504 | 0.507 | 0.059 |

AUROC = 0.50 ± 0.02 across all rates and all models. This is the expected biological outcome: with ~50 mutations across 30,000 bases, only ~25% of 150 bp reads span any variant-defining site, altering ~4% of k-mer tokens per read. This confirms rather than undermines the paper's contribution — the method is scoped to population-level distributional shift, not per-read variant assignment.

**On generalizability beyond SARS-CoV-2.** Acknowledged in §8. The architecture is pathogen-agnostic (raw nucleotides only). Pathogens with greater inter-strain diversity (influenza, norovirus) are identified as priority future validation targets.
