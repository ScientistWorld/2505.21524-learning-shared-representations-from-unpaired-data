# Problem Definition

## Research Question

Can two different data views that describe the same underlying item be mapped into a shared representation when only a very small subset of cross-view pairs is available and most training samples are unpaired or misaligned?

## Setting

Two tabular views of handwritten digits (UCI Multiple Features): 64 Karhunen-Loeve coefficients (`kar`) and 240 pixel averages (`pix`). Training data: 100 known pairs and 990 rows per view whose cross-view correspondence is unknown (each view independently thinned and shuffled). No class labels. The output is a pair of encoders, one per view, that map single rows into a common 8-dimensional space.

## Why It Matters

Paired multimodal data is often expensive or unavailable. A good solution would let researchers use abundant single-view data while requiring only a small number of known correspondences.

## Success Criteria

A successful system supports cross-view retrieval in both directions on held-out paired examples it never saw, outperforms methods trained only on the small paired subset, and stays within the fixed budget: 100 known pairs, 8 output dimensions, each view encoded independently.
