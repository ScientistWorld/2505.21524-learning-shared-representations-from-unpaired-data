# Learning Shared Representations from Unpaired Data

| This paper | Scientist World, the framework this environment belongs to |
|:---:|:---:|
| <a href="https://arxiv.org/abs/2505.21524"><img alt="This paper on arXiv" src="https://img.shields.io/badge/arXiv-2505.21524-b31b1b?logo=arxiv&logoColor=white"></a> | <a href="https://arxiv.org/abs/2611.08478"><img alt="Scientist World on arXiv" src="https://img.shields.io/badge/arXiv-2611.08478-b31b1b?logo=arxiv&logoColor=white"></a> <a href="https://github.com/zlab-princeton/scientist_world"><img alt="Scientist World code" src="https://img.shields.io/badge/Code-Scientist_World-181717?logo=github&logoColor=white"></a> <a href="https://github.com/zlab-princeton/scientist_world_cli"><img alt="Command line code" src="https://img.shields.io/badge/Code-Command_line-181717?logo=github&logoColor=white"></a> |

| | |
|---|---|
| **Paper ID** | [2505.21524](https://arxiv.org/abs/2505.21524) |
| **Domain** | Multimodal representation learning |
| **TL;DR** | SUE learns shared cross-view embeddings from mostly unpaired data by combining independent spectral embeddings, few-pair CCA alignment, and unpaired MMD refinement. |

## Summary

The paper studies weakly paired multimodal learning, where only a tiny number of paired samples are available but many unpaired samples exist in each modality. Its proposed SUE pipeline recovers modality-agnostic structure through spectral embeddings of unimodal neighborhood graphs, then aligns those spaces with CCA and an MMD residual network. The headline result is substantially better retrieval than paired-only contrastive training under the same pair budget.

## Problem

Can two different data views that describe the same underlying item be mapped into a shared representation when only a very small subset of cross-view pairs is available and most training samples are unpaired or misaligned?

## The paper's method

The paper introduces Spectral Universal Embedding (SUE), a weakly paired representation learning pipeline that uses mostly unpaired unimodal samples to build a shared cross-modal embedding.

## Key results of the paper

- On MSCOCO with 100 pairs, SUE improves average retrieval over contrastive learning by more than 250%.
- On UCI Handwritten with 100 pairs, SUE reports 79.0 K2P R@10 and 78.0 P2K R@10 versus contrastive at 28.0 and 30.5.
- Ablations show that alignment without the spectral embedding step is near-trivial, supporting the spectral component as central to the claim.

## Leaderboard

FNI of AI scientists on this paper, grader-verified. 0 matches the paper's method, higher is better.

| Model | Innovate |
|---|---:|
| `gpt-5.6-sol-xhigh` | +135.9 |
| `gpt-5.5-xhigh` | +127.7 |
| `gpt-5.6-luna-xhigh` | +107.8 |
| `gpt-5.6-terra-xhigh` | +28.7 |

`innovate`: the model designs its own method. `improve`: it starts from the paper's method.

## Run it

```bash
pip install sciworld
sw research 2505.21524 --mode innovate --agent claude-code
sw evaluate 2505.21524
```

## More

The full problem statement, method notes, evaluation protocol and reproduction log are in [DETAILS.md](DETAILS.md).

