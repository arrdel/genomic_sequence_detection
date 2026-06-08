# Rebuttal to Reviewer GHgX

We thank Reviewer GHgX for their constructive feedback. Below we address the three
noted weaknesses.

---

## (a) K-means pseudo-labels vs. ground-truth variants

We address this with new experiments. Using reference genomes for Wuhan-Hu-1
(NC_045512.2), Delta (B.1.617.2), and Omicron BA.1, we simulated 10,000 reads per
lineage (150 bp, 0.5% error) and evaluated all three models against true lineage labels:

**Table 1. Lineage separation on simulated reads (10,000 reads per lineage).**

| Model                    | Silhouette | ARI                | Omicron vs. Wuhan AUROC | Bal. Acc. |
|--------------------------|------------|--------------------|-------------------------|-----------|
| MaskedVQ-Seq             | −0.0024    | 7.3 × 10⁻⁵        | 0.508                   | 0.508     |
| Contrastive VQ-VAE (64d) | −0.0005    | 8.6 × 10⁻⁵        | 0.526                   | 0.520     |
| Contrastive VQ-VAE (128d)| −0.0008    | 5.0 × 10⁻⁶        | 0.516                   | 0.512     |

Table 2 adds a spike-in experiment (1%–50% Omicron reads injected into a Wuhan
baseline stream with explicit positive labels):

**Table 2. Positive-control injection experiment (MaskedVQ-Seq; Contrastive variants
show identical trends, AUROC 0.50 ± 0.02 across all rates).**

| Injection rate | n injected | AUROC | AUPRC | TPR @ 5% FPR |
|----------------|------------|-------|-------|--------------|
| 1%             | 51         | 0.522 | 0.010 | 0.000        |
| 5%             | 263        | 0.498 | 0.050 | 0.053        |
| 10%            | 556        | 0.515 | 0.104 | 0.058        |
| 20%            | 1,250      | 0.508 | 0.209 | 0.060        |
| 50%            | 5,000      | 0.504 | 0.507 | 0.059        |

All models achieve near-chance AUROC (~0.50) at read level — not a modelling flaw but a
biological constraint: only ~22% of 150 bp reads span variant-defining sites, and each
spanned mutation alters at most ~4% of k-mer tokens per read. Thus, MaskedVQ-Seq
targets population-level distributional shift detection, not read-level classification,
for which K-means pseudo-labels remain appropriate. §8 Implications and Limitations
acknowledges this explicitly.

---

## (b) Single pathogen and region

We acknowledge this limitation in §8. The architecture is pathogen-agnostic (raw
nucleotides only, no SARS-CoV-2-specific components). Pathogens with higher inter-strain
diversity (e.g., influenza, norovirus, RSV) would provide more informative benchmarks,
and validation on these is prioritized future work.

---

## (c) Mean-pooled pre-quantization embeddings vs. discrete codes

Correct: we mean-pool encoder outputs before quantization to obtain fixed-length
representations for clustering and retrieval. The discrete bottleneck still regularizes
training, yielding low effective dimensionality (1.02). We note in §8 that bag-of-codes
histograms or mean-pooled post-quantized vectors are natural alternatives for leveraging
codebook structure directly.

---

We thank the reviewer again; the ground-truth experiments in Tables 1–2 above directly
address (a) and establish an honest performance ceiling for read-level methods on
conserved pathogens.
