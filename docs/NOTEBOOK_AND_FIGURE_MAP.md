# Notebook and artifact map

The six public notebooks are the manuscript-facing reproduction route. They work from the fixed manifests, saved predictions and locked analysis settings in this repository; they do not launch new model training.

| Notebook | Manuscript-facing outputs |
|---|---|
| `01_benchmark_audit.ipynb` | Benchmark overlap/grouping checks, Tables 2–3 and Figure 4 |
| `02_classification.ipynb` | Classification evaluation, Table 4, Figures 5–6, exact-GC and calibration records |
| `03_activity_models.ipynb` | MPRA model comparison, Table 5, Figure 7, validation-only model selection and training-adequacy records |
| `04_external_transfer.ipynb` | External DHS-natural transfer, Table 7 and Figure 8 |
| `05_statistics.ipynb` | Table 6 / Supplementary Table S18 and the 14 headline statistical contrasts |
| `06_figures_and_tables.ipynb` | Final main/supplementary tables and figure assets from the released numeric records |

The command-line equivalents are exposed through `scripts/reproduce.py`. Optional raw-data reconstruction, checkpoint inference and retraining are documented separately in `REPRODUCTION_GUIDE.md`.
