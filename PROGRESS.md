# Progress

## What Works

- Paper analyzed and briefing files filled.
- Implemented a faithful SUE pipeline for the paper's UCI Handwritten K2P/P2K retrieval setting: independent graph spectral embeddings, CCA on 100 paired examples, and residual MMD alignment on unpaired examples.
- Implemented artifact-based evaluation that does not import from `method/`.
- Implemented a paired-only contrastive baseline to validate the scoring pipeline.
- CPU job `1711044e-d35` ran the full reproduction script and passed `python validate.py --compare`.
- CPU job `f193cea4-e7f` reran the full reproduction inside the managed container and passed `python validate.py --compare`.
- CPU job `1162da06-1a4` reran the final deterministic artifact path inside the managed container and passed `python validate.py --compare`.
- Continuation audit confirmed the implementation is the paper's SUE computation rather than a surrogate.
- Continuation revalidation on 2026-06-19 passed `python validate.py --compare`, so the reset milestone is restored to `core_claim`.
- Packaging audit on 2026-06-19 found that `scripts/run.sh` should regenerate artifacts rather than score stale local artifacts; the script now runs `scripts/reproduce.sh`, train/test evaluation, and validation in one CPU job.

## Results

Job `1162da06-1a4` produced the current validated Handwritten retrieval scores:

| Method | K2P R@1 | K2P R@5 | K2P R@10 | P2K R@1 | P2K R@5 | P2K R@10 | Mean R@10 | Embedding dim |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Contrastive baseline | 10.75 | 38.75 | 54.00 | 12.00 | 38.25 | 54.50 | 54.25 | 8 |
| SUE reproduction | 19.00 | 52.25 | 69.25 | 18.25 | 47.00 | 69.00 | 69.125 | 8 |
| SUE paper reference | 25.50 | 62.00 | 79.00 | 25.00 | 61.75 | 78.00 | 78.50 | 8 |

Job `1162da06-1a4` produced the current checked-in `scoring/scores.json`: SUE mean R@10 69.125 versus the paired-only contrastive baseline at 54.25. The reproduction is below the paper's reported SUE number but preserves the core claim: under the same 8-dimensional embedding constraint, SUE substantially improves weakly paired retrieval over a paired-only contrastive baseline.

## Remaining

- Higher milestones beyond that would require adding more Table 1 datasets or paper ablations.
- The current gym is usable for future agents to optimize the Handwritten weakly paired retrieval setting.
- The continuation audit did not identify a code or scoring issue requiring changes before closing at `core_claim`; pushing higher would require adding additional Table 1 datasets or ablations.
- Current pending CPU test job should refresh `scoring/scores.json` with all evaluator metrics from regenerated artifacts.

## Issues

- Current continuation audit found that `bash scripts/evaluate.sh artifacts` may not run on the bare login-node Python when NumPy is absent there. The dependency is declared in `environment/setup.sh`, and CPU job `1162da06-1a4` verified the evaluator in the managed container.
- Job `f193cea4-e7f` passed in the managed container. Its SUE score differed slightly from the previous artifact, so the spectral eigensolver initialization and eigenvector sign orientation were made deterministic before the next validation run.
- The paper's full image-text experiments require large image/text encoders and prepared embeddings. The first target is the paper's own small tabular Handwritten benchmark to establish the core weakly paired retrieval claim quickly.
- Job `07052327-9a9` failed only because `scores.json` included debug method names not present in `reference.json`; the method itself ran successfully.
- Job `1711044e-d35` fixed that issue and passed all validation checks.
- Job `f7e6c609-927` validated a proposed class-level mAP extension, but the resulting mAP scale did not match the paper's Appendix A.5 values. The extension was removed to avoid scoring a metric with an incompatible definition.
- The paper has an internal pair-count inconsistency: Table 1/Table 12 report 100 paired samples for Handwritten, while Appendix E.8 says CCA used 600 pairs for all datasets. This reproduction follows the table setting because it is the reported Handwritten benchmark being scored.
- The paper reports 1900 unpaired Handwritten samples in Table 12, but Appendix E.7 also says 400 paired samples are held out for evaluation and the unpaired training pool is independently thinned by 10% in each modality. This implementation follows the Appendix E.7 split/preparation literally: with 400 held out and 100 training pairs, the MMD/spectral stages see 100 paired examples plus 1350 unpaired examples per modality after thinning.

## Deviations from Paper

- The paper uses parametric SpectralNet for scalable out-of-sample spectral embedding. This reproduction uses the paper-described numeric spectral embedding plus Nyström extension on the small UCI Handwritten dataset. The core computation remains the same graph diffusion eigenvector embedding from the kNN random-walk affinity, followed by CCA and MMD; the deviation is limited to using the exact eigensolver instead of training a neural approximation to those eigenvectors.
- The Handwritten setup follows Table 1/Table 12's 100-pair weakly paired setting despite Appendix E.8's global 600-pair CCA statement.
