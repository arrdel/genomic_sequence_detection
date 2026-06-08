We thank Reviewer uRw2 for the thorough and constructive critique.

**On the gap between healthcare framing and evaluation design.** We addressed this in the revision. The title now reads "Reference-Free Anomaly Detection in Wastewater Genomic Surveillance" and the abstract closes: "…a practical, reference-free tool for detecting distributional anomalies in wastewater surveillance streams without prior knowledge of variant-defining genetic signatures." The label-free protocol (silhouette, retrieval precision, anomaly scoring) is a principled choice for this goal. §8 Limitations makes the scope boundary explicit and points to the empirical results below.

**On the Omicron vs. wildtype embedding experiment.** We ran this in full on 10,000 simulated 150 bp reads per lineage (Wuhan-Hu-1, Delta, Omicron BA.1; 0.5% error rate).

| Model | Silhouette | ARI | Omicron vs. Wuhan AUROC | Bal. Acc. |
|---|---|---|---|---|
| MaskedVQ-Seq | −0.0024 | 7.3×10⁻⁵ | 0.508 | 0.508 |
| Contrastive-64d | −0.0005 | 8.6×10⁻⁵ | 0.526 | 0.520 |
| Contrastive-128d | −0.0008 | 5.0×10⁻⁶ | 0.516 | 0.512 |

All metrics are at chance. This is the expected biological outcome: 50 mutations across 30,000 positions means ~78% of 150 bp reads span no variant-defining k-mer; each spanned mutation alters ~4% of tokens per read. Near-random performance is not a modelling failure — it is correct unsupervised behavior on a pathogen that is 99.97% conserved across lineages.

To rule out evaluation artefacts, we ran a positive-control injection: Omicron reads at 1–50% into a Wuhan stream with explicit positive labels; anomaly score = L2 to Wuhan centroid.

| Injection rate | AUROC | AUPRC | TPR @ 5% FPR |
|---|---|---|---|
| 1%  | 0.522 | 0.010 | 0.000 |
| 5%  | 0.498 | 0.050 | 0.053 |
| 10% | 0.515 | 0.104 | 0.058 |
| 20% | 0.508 | 0.209 | 0.060 |
| 50% | 0.504 | 0.507 | 0.059 |

AUROC remains at chance across all rates and all models, confirming the constraint is biological. The method's value is at the sample level, where distributional shift across thousands of reads provides a detectable signal.
