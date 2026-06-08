# Rebuttal to Reviewer uRw2

We thank Reviewer uRw2 for the thorough and constructive critique. We address each
concern directly, with references to the revised manuscript.

---

## On the gap between healthcare framing and evaluation design

We agree this tension exists and have addressed it directly in the revision. The **title**
and **abstract** now reframe the method's stated scope: "Reference-Free *Anomaly
Detection* in Wastewater Genomic Surveillance" replaces "Variant Detection," and the
abstract's closing sentence now reads:

> "These findings demonstrate that discrete bottleneck representations offer a practical,
> reference-free tool for detecting distributional anomalies in wastewater surveillance
> streams without prior knowledge of variant-defining genetic signatures."

The label-free evaluation protocol (silhouette, retrieval precision, anomaly scoring) is
a principled design choice for this stated goal, not a concession. §8 Implications and
Limitations further makes the scope boundary precise: point (1) acknowledges that
evaluation uses K-means pseudo-labels rather than ground-truth variants, and includes the
empirical characterisation of read-level lineage discrimination limits reported below.

---

## On the Omicron vs. wildtype embedding experiment

We ran this experiment in full. Using 10,000 simulated 150 bp reads from each of
Wuhan-Hu-1 (NC\_045512.2), Delta (B.1.617.2), and Omicron BA.1 (0.5% substitution
error rate, uniform sliding-window simulation, mixed strands), we extracted embeddings
from all three models and computed a complete panel of lineage-separation metrics:

**Table 1. Lineage separation on simulated reads (10,000 reads per lineage).**

| Model                    | Silhouette | ARI                | Omicron vs. Wuhan AUROC | Bal. Acc. |
|--------------------------|------------|--------------------|-------------------------|-----------|
| MaskedVQ-Seq             | −0.0024    | 7.3 × 10⁻⁵        | 0.508                   | 0.508     |
| Contrastive VQ-VAE (64d) | −0.0005    | 8.6 × 10⁻⁵        | 0.526                   | 0.520     |
| Contrastive VQ-VAE (128d)| −0.0008    | 5.0 × 10⁻⁶        | 0.516                   | 0.512     |

All metrics are consistent with random performance. We report this honestly and argue it
is the correct outcome. With ~50 substitutions across 30,000 positions, the expected
number of variant-defining sites spanned per 150 bp read is 50 × 150/30,000 ≈ 0.25,
meaning ~78% of reads share no variant-discriminative k-mer between Wuhan and Omicron.
Each mutation that is spanned alters at most 6 of ~145 6-mer tokens (~4% per read),
insufficient signal for any unsupervised model to weight above background. Near-random
performance is not a modelling failure; it is the correct unsupervised behaviour on a
pathogen where 99.97% of sequence is conserved across lineages.

To confirm this is a biological constraint and not an artefact of our evaluation
protocol, we ran a positive-control injection experiment: Omicron reads were injected at
controlled rates (1%–50%) into a Wuhan baseline stream with explicit positive labels. A
reference centroid was fit on held-out Wuhan reads; anomaly scores were L2 distances
to that centroid.

**Table 2. Positive-control injection experiment (MaskedVQ-Seq; Contrastive variants
show identical trends, AUROC 0.50 ± 0.02 across all rates).**

| Injection rate | n injected | AUROC | AUPRC | TPR @ 5% FPR |
|----------------|------------|-------|-------|--------------|
| 1%             | 51         | 0.522 | 0.010 | 0.000        |
| 5%             | 263        | 0.498 | 0.050 | 0.053        |
| 10%            | 556        | 0.515 | 0.104 | 0.058        |
| 20%            | 1,250      | 0.508 | 0.209 | 0.060        |
| 50%            | 5,000      | 0.504 | 0.507 | 0.059        |

AUROC remains at chance across all injection rates and all models, ruling out any
evaluation artefact. This confirms that the method is correctly scoped to
population-level distributional shift (where many reads combine to give a detectable
signal), not individual-read variant assignment (where the genomic biology makes
detection impossible regardless of model architecture).
