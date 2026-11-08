# Experiment Overview

## handwritten_retrieval

Tests whether a compact (8-dimensional) shared representation, learned from many unpaired single-view samples and only 100 known cross-view pairs, can recover the corresponding item between two tabular views of handwritten digits it has never seen. Each method supplies one encoder per view; the evaluator encodes 400 held-out items per view independently, in its own order, and measures how highly each item's true counterpart is ranked among all 400 candidates, in both retrieval directions. The paired-only baseline shows what the 100 pairs alone achieve; the gap above it is the value extracted from unpaired data. This is the paper's Handwritten retrieval benchmark (Table 1), scaled to a smaller unpaired pool so that a public development split can be held out.
