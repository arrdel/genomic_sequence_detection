# Rebuttal to Reviewer GHgX

We thank Reviewer GHgX for their constructive feedback. Below we address the three
noted weaknesses.

---

## (a) K-means pseudo-labels vs. ground-truth variants

We address this with new experiments in **Appendix D**. Using reference genomes for
Wuhan-Hu-1, Delta, and Omicron BA.1, we simulated 10,000 reads per lineage (150 bp,
0.5% error) and evaluated against true labels (**Table D.1**: silhouette, ARI, NMI,
AUROC). **Table D.2** adds a spike-in experiment (1–50% Omicron reads into a Wuhan
baseline stream with explicit positive labels). All models achieve near-chance AUROC
(~0.50) at read level — not a modelling flaw but a biological constraint: only ~22% of
150 bp reads span variant-defining sites. Thus, MaskedVQ-Seq targets population-level
distributional shift detection, not read-level classification, for which K-means
pseudo-labels remain appropriate. **§8 Implications and Limitations** now directs
readers to Appendix D for the full empirical characterisation of these read-level limits.

---

## (b) Single pathogen and region

We acknowledge this limitation in **§8**. The architecture is pathogen-agnostic (raw
nucleotides only, no SARS-CoV-2-specific components). Pathogens with higher inter-strain
diversity (e.g., influenza, norovirus, RSV) would provide more informative benchmarks,
and validation on these is prioritised future work.

---

## (c) Mean-pooled pre-quantization embeddings vs. discrete codes

Correct: we mean-pool encoder outputs before quantization to obtain fixed-length
representations for clustering and retrieval. The discrete bottleneck still regularises
training, yielding low effective dimensionality (1.02). We now note in **§8** that
bag-of-codes histograms or mean-pooled post-quantised vectors are natural alternatives
for leveraging codebook structure directly.

---

We thank the reviewer again; the ground-truth experiments in **Appendix D** directly
address (a) and strengthen the paper by establishing an honest performance ceiling for
read-level methods on conserved pathogens.
