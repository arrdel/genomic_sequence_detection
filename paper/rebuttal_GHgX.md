We thank Reviewer GHgX for the constructive feedback and address the three weaknesses.

**(a) K-means pseudo-labels vs. ground-truth variants.** We ran new experiments using reference genomes for Wuhan-Hu-1, Delta, and Omicron BA.1 (10,000 simulated 150 bp reads per lineage, 0.5% error) and evaluated all models against true lineage labels:

| Model | Silhouette | ARI | Omicron vs. Wuhan AUROC | Bal. Acc. |
|---|---|---|---|---|
| MaskedVQ-Seq | −0.0024 | 7.3×10⁻⁵ | 0.508 | 0.508 |
| Contrastive-64d | −0.0005 | 8.6×10⁻⁵ | 0.526 | 0.520 |
| Contrastive-128d | −0.0008 | 5.0×10⁻⁶ | 0.516 | 0.512 |

A spike-in experiment (1–50% Omicron into a Wuhan stream with explicit positive labels) further confirms the result:

| Injection rate | AUROC | AUPRC | TPR @ 5% FPR |
|---|---|---|---|
| 1%  | 0.522 | 0.010 | 0.000 |
| 5%  | 0.498 | 0.050 | 0.053 |
| 10% | 0.515 | 0.104 | 0.058 |
| 20% | 0.508 | 0.209 | 0.060 |
| 50% | 0.504 | 0.507 | 0.059 |

All models achieve near-chance AUROC (~0.50) — a biological constraint, not a modelling flaw: only ~22% of 150 bp reads span variant-defining sites between Wuhan and Omicron. MaskedVQ-Seq therefore targets population-level shift detection, for which K-means pseudo-labels are appropriate. §8 Limitations acknowledges this explicitly.

**(b) Single pathogen and region.** Acknowledged in §8. The architecture requires only raw nucleotides (no SARS-CoV-2-specific components). Validation on higher-diversity pathogens (influenza, norovirus, RSV) is prioritized future work.

**(c) Mean-pooled pre-quantization embeddings vs. discrete codes.** The discrete bottleneck regularizes training (effective dimensionality 1.02) even though we use pre-quantization mean pooling for downstream tasks. §8 notes that bag-of-codes histograms or post-quantization pooling are natural next steps.
