# SpecDiff

Official code repository for spectral generation experiments comparing `cVAE`, `cWGAN-GP`, and `SpecDiff`.

## Repository Layout

```text
SpecDiff/
├── README.md
├── requirements.txt
├── data/
├── scripts/
│   ├── train_cvae.py
│   ├── train_cwgan_gp.py
│   ├── train_specdiff.py
│   ├── train_classifier.py
│   ├── evaluate_models.py
│   ├── evaluate_generated_data.py
│   ├── generate_cvae.py
│   ├── generate_cwgan_gp.py
│   ├── generate_specdiff.py
│   └── sample_specdiff.py
└── src/
    ├── data.py
    ├── metrics.py
    ├── real_fake_classifier.py
    ├── utils.py
    ├── models/
    │   ├── cvae.py
    │   ├── cwgan_gp.py
    │   └── specdiff.py
    ├── training/
    │   ├── cvae.py
    │   ├── cwgan_gp.py
    │   ├── specdiff.py
    │   └── classifier.py
    ├── evaluation/
    │   ├── model_comparison.py
    │   ├── generated_data.py
    │   └── plots.py
    └── generation/
        ├── cvae.py
        ├── cwgan_gp.py
        ├── specdiff.py
        └── specdiff_sampling.py
```

## Directory Summary

- `scripts/` contains runnable experiment entry points.
- `src/models/` contains model definitions for `cVAE`, `cWGAN-GP`, and `SpecDiff`.
- `src/training/` contains training pipelines for generative models and the downstream classifier.
- `src/evaluation/` contains model-comparison and generated-data evaluation code.
- `src/generation/` contains synthetic-data export and class-wise sampling utilities.
- `data/` is reserved for datasets and split files and is not tracked by default.
