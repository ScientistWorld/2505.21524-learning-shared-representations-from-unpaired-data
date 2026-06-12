# Overview

- **Paper ID:** 2505.21524
- **Title:** Learning Shared Representations from Unpaired Data
- **Domain:** Multimodal representation learning
- **TL;DR:** SUE learns shared cross-view embeddings from mostly unpaired data by combining independent spectral embeddings, few-pair CCA alignment, and unpaired MMD refinement.

## Short Summary

The paper studies weakly paired multimodal learning, where only a tiny number of paired samples are available but many unpaired samples exist in each modality. Its proposed SUE pipeline recovers modality-agnostic structure through spectral embeddings of unimodal neighborhood graphs, then aligns those spaces with CCA and an MMD residual network. The headline result is substantially better retrieval than paired-only contrastive training under the same pair budget.

## Key Results

- On MSCOCO with 100 pairs, SUE improves average retrieval over contrastive learning by more than 250%.
- On UCI Handwritten with 100 pairs, SUE reports 79.0 K2P R@10 and 78.0 P2K R@10 versus contrastive at 28.0 and 30.5.
- Ablations show that alignment without the spectral embedding step is near-trivial, supporting the spectral component as central to the claim.
