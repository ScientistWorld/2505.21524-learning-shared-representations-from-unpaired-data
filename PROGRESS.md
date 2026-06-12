# Progress

## What Works

- Paper analyzed and briefing files filled.
- Implemented a faithful SUE pipeline for the paper's UCI Handwritten K2P/P2K retrieval setting: independent graph spectral embeddings, CCA on 100 paired examples, and residual MMD alignment on unpaired examples.
- Implemented artifact-based evaluation that does not import from `method/`.
- Implemented a paired-only contrastive baseline to validate the scoring pipeline.
- CPU job `1711044e-d35` ran the full reproduction script and passed `python validate.py --compare`.
- Continuation audit confirmed the implementation is the paper's SUE computation rather than a surrogate. The evaluator is being extended to score the paper's class-level mAP metric from the same saved retrieval artifacts.

## Results

Job `1711044e-d35` produced validated Handwritten retrieval scores:

| Method | K2P R@1 | K2P R@5 | K2P R@10 | P2K R@1 | P2K R@5 | P2K R@10 | Mean R@10 | Embedding dim |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Contrastive baseline | 10.75 | 38.75 | 54.00 | 12.00 | 38.25 | 54.50 | 54.25 | 8 |
| SUE reproduction | 17.50 | 50.50 | 69.25 | 17.50 | 47.75 | 69.75 | 69.50 | 8 |
| SUE paper reference | 25.50 | 62.00 | 79.00 | 25.00 | 61.75 | 78.00 | 78.50 | 8 |

The reproduction is below the paper's reported SUE number but preserves the core claim: under the same 8-dimensional embedding constraint, SUE substantially improves weakly paired retrieval over a paired-only contrastive baseline.

## Remaining

- The mAP scoring extension needs a packaged-environment validation job before it can be counted as additional reproduced coverage.
- Higher milestones beyond that would require adding more Table 1 datasets or paper ablations.
- The current gym is usable for future agents to optimize the Handwritten weakly paired retrieval setting.

## Issues

- The paper's full image-text experiments require large image/text encoders and prepared embeddings. The first target is the paper's own small tabular Handwritten benchmark to establish the core weakly paired retrieval claim quickly.
- Job `07052327-9a9` failed only because `scores.json` included debug method names not present in `reference.json`; the method itself ran successfully.
- Job `1711044e-d35` fixed that issue and passed all validation checks.

## Deviations from Paper

- The paper uses parametric SpectralNet for scalable spectral embedding. This reproduction uses the paper-described numeric spectral embedding plus Nyström extension on the small UCI Handwritten dataset. The core computation remains the same graph diffusion eigenvector embedding; this avoids unnecessary neural approximation for a 2000-sample tabular benchmark.
