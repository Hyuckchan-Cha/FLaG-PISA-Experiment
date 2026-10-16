# FLaG-PISA-Experiment

Fitting the **Fused Latent and Graphical IRT model** (FLaG-IRT; Chen et al., 2018) to the PISA 2018 science assessment, and comparing it with exploratory VAE-based IRT models. (a VAE-based IRT model with an item graphical structure is called VIF-model here)

The model explains item responses with a **latent ability (θ)** plus a **graph S of direct item-to-item links** (local dependence). The notebook estimates both on real large-scale assessment data (2,711 students × 102 items, 71% missing) and asks how the estimated graph relates to the known item clusters.

## Models

| Model | Handling of θ | Role |
|---|---|---|
| **FLaG-IRT** | integrated out (pseudo-likelihood + L1 penalty on S) | established model |
| **FLaG-nograph** | same, with S = 0 | warm start for FLaG |
| **VIF-model** | estimated by an encoder (amortized variational inference) | experimental VAE-based variant with the same graph term |
| **MIRT-only** (VAE-MIRT) | same, with S = 0 | graph-free baseline |

Models are compared on held-out prediction (AUC, NLL per cell) and on the estimated graph (edges selected by BIC, agreement with the item clusters).

## Things to know

- **The data is not included.** The notebook expects a preprocessed CSV at `data/modified data(science 2018).csv` (rows = 102 items, columns = students, values 0/1, missing = empty). The original data is available from the OECD PISA 2018 database. If you need the preprocessed data or details on how it was prepared, please contact me by email (see below).
- **The notebook is saved without outputs.** Run it to reproduce the tables and figures.
- **Graph metrics are not ground truth.** There is no true graph for real data; "precision" means agreement with the PISA item clusters, which also overlap with the booklet design. The notebook includes a co-observation diagnostic for this reason.
- **This is ongoing research.** The VIF-model is exploratory, and results may change.

## How to run

```bash
pip install torch numpy pandas matplotlib joblib
```

- Python 3.10+, CPU only.
- Set `DATA_PATH` in the first code cell. Try `SMOKE = True` first to check that everything runs end to end.
- A full run takes about 35 min (VIF-model) + 25 min (FLaG) per split on 32 cores. Set the number of workers with the `IRT_WORKERS` environment variable.

## References

- Chen, Y., Li, X., Liu, J., & Ying, Z. (2018). Robust measurement via a fused latent and graphical item response theory model. *Psychometrika, 83*(3), 538–562.
- OECD (2019). *PISA 2018 Database*.

## Author & Contact

Hyuckchan Cha · Seoul National University · [hyuckchan-cha.github.io](https://hyuckchan-cha.github.io)

For the data or data preprocessing details, please email chahchan10039@snu.ac.kr.
