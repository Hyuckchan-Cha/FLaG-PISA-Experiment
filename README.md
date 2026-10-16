# FLaG-PISA-Experiment

Fitting the **Fused Latent and Graphical IRT model** (FLaG-IRT; Chen et al., 2018) to the PISA 2018 science assessment, and comparing it with an exploratory VAE-based IRT model that has the same item graph (called **VIF-model** here).

Both models explain item responses with a **latent ability (θ)** plus a **graph S of direct item-to-item links** (local dependence). The notebook estimates both on real large-scale assessment data (2,711 students × 102 items, 71% missing) and asks how the estimated graph relates to the known item clusters.

## Models

All responses are binary, $x_{ij}\in\{0,1\}$ (student $i$, item $j$). $S$ is a symmetric item × item matrix with zero diagonal; $s_{jk}\neq 0$ means an edge between items $j$ and $k$.

| Model | Equation | Handling of θ | Role |
|---|---|---|---|
| **FLaG-IRT** | $\mathrm{logit}\,P(x_{ij}=1\mid x_{i,-j})=\sum_{k\neq j}M_{jk}x_{ik}+M_{jj}/2$, with $M=A\Sigma A^\top+S$ | integrated out | established model |
| **FLaG-nograph** | FLaG-IRT with off-diagonal $S=0$ | integrated out | warm start for FLaG |
| **VIF-model** | $\mathrm{logit}\,P(x_{ij}=1)=a_j\theta_i-d_j+\sum_{k\neq j}s_{jk}x_{ik}$ | estimated by an encoder | experimental model |
| **MIRT-only** (VAE-MIRT) | VIF-model with $S=0$ | estimated by an encoder | graph-free baseline, warm start for VIF |

- **FLaG-IRT**: integrating θ out turns the model into an Ising-type model with item interaction matrix $M$. It is fitted by pseudo-likelihood (the sum of each item's conditional log-likelihood; Besag, 1975).
- **VIF-model**: the same graph term added to a VAE-based IRT model (Curi et al., 2019; Wu et al., 2020; Urban & Bauer, 2021). An encoder (two-layer tanh MLP) reads each student's responses and outputs $q(\theta_i\mid x_i)=N(\mu_i,\sigma_i^2)$, with prior $N(0,1)$. θ is therefore defined explicitly for each student, unlike in FLaG-IRT.

## Fitting

Both models are fitted the same way, so that differences come from the models rather than the optimizer:

- **Adam + plain gradient step**: the smooth parameters are updated with Adam (Kingma & Ba, 2015), while the edge weights $s$ take a plain gradient step followed by soft-thresholding for the L1 penalty (Parikh & Boyd, 2014), so edges can become exactly zero.
- **Graph selection**: the support of $S$ is chosen by **BIC** along a path of L1 penalties, following Chen et al. (2018).
- **Evaluation**: AUC and NLL on held-out cells (85/15 train/test split, repeated over several splits), and agreement of the selected edges with the item clusters.

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

The full reference list is in the first cell of the notebook. Key references:

- Chen, Y., Li, X., Liu, J., & Ying, Z. (2018). Robust measurement via a fused latent and graphical item response theory model. *Psychometrika, 83*(3), 538–562.
- Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic optimization. *ICLR*.
- Parikh, N., & Boyd, S. (2014). Proximal algorithms. *Foundations and Trends in Optimization, 1*(3).
- Urban, C. J., & Bauer, D. J. (2021). A deep learning algorithm for high-dimensional exploratory item factor analysis. *Psychometrika, 86*(1), 1–29.
- OECD (2019). *PISA 2018 Database*.

## Author & Contact

Hyuckchan Cha · Seoul National University · [hyuckchan-cha.github.io](https://hyuckchan-cha.github.io)

For the data or data preprocessing details, please email chahchan10039@snu.ac.kr.
