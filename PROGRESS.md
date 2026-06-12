# Progress

## What Works

- Paper analyzed and briefing files filled.
- Implemented a faithful SUE pipeline for the paper's UCI Handwritten K2P/P2K retrieval setting: independent graph spectral embeddings, CCA on 100 paired examples, and residual MMD alignment on unpaired examples.
- Implemented artifact-based evaluation that does not import from `method/`.
- Implemented simple baselines to validate the scoring pipeline.

## Results

No reproduced job results yet. The first submitted job will run `scripts/reproduce.sh`, write `scoring/scores.json`, and compare against the paper's Handwritten Table 1 numbers.

## Remaining

- Run the first compute job to verify the container, method, baselines, and evaluator.
- If the Handwritten result supports the core claim, update milestones to `core_claim`; otherwise diagnose and iterate.
- Add a second dataset or ablation if budget remains.

## Issues

- The paper's full image-text experiments require large image/text encoders and prepared embeddings. The first target is the paper's own small tabular Handwritten benchmark to establish the core weakly paired retrieval claim quickly.

## Deviations from Paper

- The paper uses parametric SpectralNet for scalable spectral embedding. This reproduction uses the paper-described numeric spectral embedding plus Nyström extension on the small UCI Handwritten dataset. The core computation remains the same graph diffusion eigenvector embedding; this avoids unnecessary neural approximation for a 2000-sample tabular benchmark.
