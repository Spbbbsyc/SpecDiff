# Code Structure

spectra/
├── README.md
├── requirements.txt
├── checkpoints/
│   ├── specdiff/
│   │   ├── best.pt
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
    ├── __init__.py
    ├── data.py
    ├── metrics.py
    ├── real_fake_classifier.py
    ├── utils.py
    ├── models/
    │   ├── __init__.py
    │   ├── cvae.py
    │   ├── cwgan_gp.py
    │   └── specdiff.py
    ├── training/
    │   ├── __init__.py
    │   ├── cvae.py
    │   ├── cwgan_gp.py
    │   ├── specdiff.py
    │   └── classifier.py
    ├── evaluation/
    │   ├── __init__.py
    │   ├── model_comparison.py
    │   ├── generated_data.py
    │   └── plots.py
    └── generation/
        ├── __init__.py
        ├── cvae.py
        ├── cwgan_gp.py
        ├── specdiff.py
        └── specdiff_sampling.py
```

- `scripts/` contains runnable entry-point scripts.
- `models/` corresponds to the three paper models: `cVAE`, `cWGAN-GP`, and `SpecDiff`.
- `training/` contains model training pipelines and the downstream classifier training pipeline.
- `evaluation/` contains model-comparison and generated-data evaluation code.
- `generation/` contains synthetic-data export and class-wise sampling code.
