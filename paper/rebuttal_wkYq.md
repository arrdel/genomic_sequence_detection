# Rebuttal to Reviewer wkYq

We thank Reviewer wkYq for the careful reading and constructive tone.

---

## On venue fit and clinical relevance

We respectfully push back. MLHC explicitly includes a Novel Methods theme covering
unsupervised representation learning for healthcare data; Reviewer GHgX specifically
noted this paper is well-suited to that theme. Wastewater-based epidemiology directly
informed public health decisions at the population level throughout COVID-19, with
surveillance programs now operational across the US, EU, and internationally. Our method
sits at the specific point in that pipeline where novel variants would otherwise go
undetected. The revised title and abstract now make the operational framing explicit: we
provide upstream flagging infrastructure that determines which samples get sent for
targeted clinical sequencing, not a clinical decision tool itself.

---

## On the anomaly detection experimental design

We agree and have now run the experiment the reviewer describes. We injected Omicron
reads at controlled rates (1%–50%) into a Wuhan baseline stream with explicit positive
labels. A reference centroid was fit on held-out Wuhan reads; anomaly scores were L2
distances to that centroid.

**Table 1. Positive-control injection experiment (MaskedVQ-Seq; Contrastive variants
show identical trends, AUROC 0.50 ± 0.02 across all rates).**

| Injection rate | n injected | AUROC | AUPRC | TPR @ 5% FPR |
|----------------|------------|-------|-------|--------------|
| 1%             | 51         | 0.522 | 0.010 | 0.000        |
| 5%             | 263        | 0.498 | 0.050 | 0.053        |
| 10%            | 556        | 0.515 | 0.104 | 0.058        |
| 20%            | 1,250      | 0.508 | 0.209 | 0.060        |
| 50%            | 5,000      | 0.504 | 0.507 | 0.059        |

AUROC is 0.50 ± 0.02 across all rates and all models; AUPRC tracks the random baseline.
This outcome is consistent with a biological constraint: with ~50 mutations across
30,000 bases, fewer than 25% of 150 bp reads span any variant-defining site, and those
reads have only ~4% of their k-mer tokens altered. §8 Implications and Limitations makes
this explicit. Critically, this result confirms rather than undermines the paper's
contribution: the method is correctly scoped to population-level distributional shift
(where many reads combine to give a detectable signal), not individual-read variant
assignment (where the genomic biology makes detection impossible regardless of model
architecture).

---

## On generalizability beyond SARS-CoV-2

This remains the most significant limitation and is explicitly stated in §8 "Implications
and limitations." The architecture is pathogen-agnostic; it requires only raw nucleotide
sequences and makes no SARS-CoV-2-specific assumptions. Pathogens with greater
inter-strain diversity (influenza H3N2 vs. H1N1, norovirus GI vs. GII) would also
provide more informative lineage-separation benchmarks, since a larger fraction of
150 bp reads would span variant-defining sites. Validation on these targets is identified
as priority future work.

---

## On the synthetic data section

[Unchanged from prior submission — reviewer did not raise new concerns on this point.]
