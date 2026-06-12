# Progress

## What Works

- Paper analyzed and briefing files filled.
- Implemented a faithful SUE pipeline for the paper's UCI Handwritten K2P/P2K retrieval setting: independent graph spectral embeddings, CCA on 100 paired examples, and residual MMD alignment on unpaired examples.
- Implemented artifact-based evaluation that does not import from `method/`.
- Implemented simple baselines to validate the scoring pipeline.
- First CPU job ran the method end to end and produced SUE retrieval artifacts.

## Results

Job `07052327-9a9` produced SUE Handwritten retrieval scores before failing validation on extra debug baseline names: K2P R@10 68.5, P2K R@10 69.25, mean R@10 68.875. This is below the paper's reported mean R@10 78.5 but substantially above random and the original-space CCA debug baseline from that run.

## Remaining

- Rerun with a paper-named contrastive baseline so `validate.py --compare` accepts `scores.json`.
- If the accepted run preserves the SUE-vs-contrastive gap, update milestones to `core_claim`.
- Add a second dataset or ablation if budget remains.

## Issues

- The paper's full image-text experiments require large image/text encoders and prepared embeddings. The first target is the paper's own small tabular Handwritten benchmark to establish the core weakly paired retrieval claim quickly.
- Job `07052327-9a9` failed only because `scores.json` included debug method names not present in `reference.json`; the method itself ran successfully.

## Deviations from Paper

- The paper uses parametric SpectralNet for scalable spectral embedding. This reproduction uses the paper-described numeric spectral embedding plus Nyström extension on the small UCI Handwritten dataset. The core computation remains the same graph diffusion eigenvector embedding; this avoids unnecessary neural approximation for a 2000-sample tabular benchmark.
