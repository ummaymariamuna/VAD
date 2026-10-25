# Repository Organization Notes

Temporary working document for organizing the repository. Delete this file after the cleanup is complete.

## Current Python organization

The active Python research code is now organized under `src/` and remains directly runnable with hardcoded values:

```text
src/
├── requirements.txt
├── ten_second_split.py
├── video_augmentation.py
├── video_embedding_cnn.py
├── video_embedding_resnet.py
├── video_embedding_vit.py
├── sru_training.py
├── srupp_training.py
├── video_clustering.py
├── video_inference.py
├── real_time_vad_config.py
├── real_time_vad_processor.py
└── real_time_vad_usage.py
```

Supporting Python files are separated into `tests/test_real_time_buffer.py` and `archive/window.py`. Duplicate Python versions from `utils/utils/` were removed. No CLI or argument parser was added; edit the configuration constants in each file before running it.

This reorganization was intentionally not verified or executed.

## Canonical research story

- Published paper: **“Vision transformer embedded video anomaly detection using attention driven recurrence”**
- Venue: *Array*, Volume 27 (2025), Article 100471
- DOI: <https://doi.org/10.1016/j.array.2025.100471>
- Main pipeline:
  1. Segment and manually label surveillance videos.
  2. Reduce frames and apply augmentation.
  3. Extract 2048-dimensional frame features with ViT-Small.
  4. Model temporal information with a two-layer, eight-head SRU++.
  5. Apply cluster-based score weighting for UCF-Crime multiclass prediction.
  6. Use a sliding window for inference on long videos.
- Evaluated datasets: UCF-Crime, RWF-2000, and SCVD.
- Published test results:

| Dataset | Accuracy | AUC |
| --- | ---: | ---: |
| UCF-Crime | 95.14% | 0.96 |
| RWF-2000 | 92.88% | 0.95 |
| SCVD | 93.73% | 0.93 |

The paper is the canonical description. The current `README.md` reflects an earlier ResNet-focused version and should eventually be aligned with the published ViT + SRU++ method.

## Repository map before Python reorganization

```text
.
├── README.md                  # Short, earlier project description
├── paper.pdf                  # Published paper
├── LICENSE                    # MIT license
├── requirements.txt           # Partial Python dependencies
├── base-dataset/
│   └── README.md              # Original UCF-Crime source and classes
├── dataset/
│   └── README.md              # Preprocessing and labeling notes
├── experiments/               # 22 Kaggle/Jupyter research notebooks
│   ├── embeddings-*.ipynb     # CNN/ResNet embedding experiments
│   ├── SRU-*.ipynb            # Binary SRU experiments
│   ├── SRUpp-*.ipynb          # Binary SRU++ experiments
│   ├── multiclass-*.ipynb     # Multiclass SRU++ training
│   ├── working.ipynb          # Cluster-weighting experiment
│   └── test-*.ipynb           # Full-video multiclass inference test
└── utils/
    ├── ten_second_split.*     # Video segmentation
    ├── video_augmentation.py  # Dataset augmentation
    ├── video_embedding_*.py   # CNN, ResNet, and ViT embeddings
    ├── sru*_training.py       # SRU and SRU++ training
    ├── video_clustering.py    # Cluster centers and similarity
    ├── video_inference.py     # Batch inference prototype
    ├── real_time_vad_*.py     # Buffered stream inference prototype
    ├── test_real_time_buffer.py
    ├── window.py              # Discarded low-accuracy approach
    ├── requirements.txt       # Duplicate of root requirements
    └── utils/                 # Older duplicate preprocessing files
```

## What is already valuable evidence

- `paper.pdf` is the strongest research artifact.
- Executed notebooks preserve experiment configuration, training logs, plots, and results.
- The SRU/ResNet notebook series records comparisons at multiple epoch counts.
- `multiclass-training-sru.ipynb` contains the 12-class SRU++ experiment and per-class ROC/PR work.
- `working.ipynb` records the feature-cluster weighting idea used in the paper.
- Dataset notes document class removal, ten-second segmentation, manual labeling, and balancing.
- The utility scripts show work toward reusable preprocessing, training, batch inference, and real-time inference.

The notebooks are research records, not yet a clean reproducible pipeline. Their outputs should be preserved until important results have been extracted into documentation.

## Duplication and archive candidates

- `experiments/sru-with-e-cnn-v1-100-2048.ipynb` is identical in source and outputs to `SRU-CNNv1-100-2048-e200.ipynb`.
- `experiments/srupp-with-e-cnn-v1-100-2048.ipynb` is identical to `SRUpp-CNNv1-100-2048-e200.ipynb`.
- `utils/ten_second_split.ipynb` and `utils/utils/ten_second_split.ipynb` are identical.
- `utils/utils/` contains older variants of the top-level preprocessing files.
- `window.py` explicitly documents a discarded approach and belongs in an archive, not the main pipeline.
- `notebook71c182508e.ipynb` is an unfinished ResNet experiment ending in a CUDA out-of-memory error.

Do not delete these until provenance and useful outputs have been checked.

## Original Python review queue

1. Fix module/class-name mismatches between embedding scripts, inference, and real-time processing.
2. Fix the missing argument-parser initialization in `srupp_training.py`.
3. Make embedding dimensions, frame counts, model types, class counts, and checkpoint formats consistent.
4. Reconcile cluster model saving/loading with batch and real-time inference.
5. Correct training metric accumulation and separate binary from multiclass evaluation.
6. Replace hard-coded Kaggle paths with command-line/config inputs.
7. Complete dependencies (`torchvision`, `timm`, plotting packages, and others actually used).
8. Convert the buffer checks into normal automated tests.
9. Preserve the paper parameters separately from exploratory alternatives.

All Python files currently parse as valid Python, but the pipeline has not been executed because datasets, embeddings, model checkpoints, and some dependencies are not present.

## Earlier proposed target structure

This is a proposed layout only; no files have been moved.

```text
.
├── README.md
├── CITATION.cff
├── LICENSE
├── requirements.txt
├── paper/
│   └── paper.pdf
├── docs/
│   ├── dataset.md
│   └── results.md
├── src/video-anomaly-detection/
│   ├── preprocessing.py
│   ├── embeddings.py
│   ├── models.py
│   ├── clustering.py
│   └── inference.py
├── scripts/
│   ├── prepare_data.py
│   ├── train.py
│   └── infer.py
├── experiments/
│   ├── paper/
│   └── archive/
└── tests/
```

Large datasets, embeddings, checkpoints, logs, and generated outputs should stay outside Git and be documented with download or generation instructions.

## Safe cleanup order

1. Preserve the current working tree and decide which untracked research files should be committed.
2. Repair and consolidate the Python pipeline.
3. Select the canonical paper experiments; archive duplicates and failed trials.
4. Add tests and one small smoke-test path.
5. Rewrite `README.md` around the published paper, contributions, results, setup, and citation.
6. Add `CITATION.cff`, environment details, and links to data/model artifacts.

## Current working-tree warning

Before this note was created, `utils/ten_second_split.py` and `utils/window.py` already had local modifications, and the paper, many notebooks, and most newer utility scripts were untracked. Preserve that work during reorganization.
