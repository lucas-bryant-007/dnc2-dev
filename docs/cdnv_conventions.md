# CDNV normalization contract

This repository uses one explicit internal convention and converts at paper
interfaces. A bare symbol `V` or an unlabeled `Vij` must not be added to a new
artifact.

Primary references: [original CDNV (ICLR 2022)](https://arxiv.org/pdf/2112.15121),
[2025 directional bound](https://arxiv.org/html/2506.04411), and
[2026 directional restatement](https://arxiv.org/html/2603.03530).

For class trace variances (v_i,v_j) and squared mean gap (d_{ij}^2):

| Name in code | Definition | Provenance |
|---|---:|---|
| `ordered_single_class` | (v_i/d_{ij}^2) | 2025 paper's main definition |
| `original_half_symmetric` | ((v_i+v_j)/(2d_{ij}^2)) | original ICLR 2022 CDNV |
| `unhalved_symmetric` | ((v_i+v_j)/d_{ij}^2) | current B-theorems and 2026 pairwise schema |

The canonical internal B-theorem convention is `unhalved_symmetric`. Therefore


\[
V_B=\frac{r-B}{2B},\qquad
\widetilde V_B=\frac{1-B}{2B}.
\]

In the original half-normalization, the same identities are

\[
V_{\mathrm{original}}=\frac{r-B}{4B},\qquad
\widetilde V_{\mathrm{original}}=\frac{1-B}{4B}.
\]

`analysis.cdnv_conventions` is the sole conversion surface. The legacy
pairwise key `Vij` remains readable but is explicitly tagged
`unhalved_symmetric`; new outputs also include `Vij_original_half_symmetric`.
`Vtilde_ij` is ordered single-class.

## 2025 comparison curves

The two comparison curves are deliberately separate:

- `luthra2025_a16_published` is the displayed (a=16) Proposition 1
  corollary under the paper's defined-symbol convention, using the ordered
  metrics and
  (V_f^s=\operatorname{Avg}_{i\ne j}\sqrt{v_i/d_{ij}^2}).
- `luthra2025_optimized_official` ports the authors' Cardano optimizer from
  `DLFundamentals/directional-nc` commit
  `947f1410e12034a5a6097bf2884040110cc1b8c7`. It intentionally uses
  (sqrt{V_f}), matching their code, and population class variances with
  denominator (N).

The backward-compatible output field `luthra2025` now aliases the official
optimized curve. It must not be interpreted as the fixed-(a) curve.

The 2025 proof text itself replaces its earlier ordered (V_{ij}=v_i/d_{ij}^2)
with the unhalved symmetric ((v_i+v_j)/d_{ij}^2) during the derivation. The
`a16_published` label above means the displayed theorem interpreted using its
declared symbols; the official curve is kept separate precisely because the
paper and reference implementation are not normalization-identical.

## Validity contract

- 2025 Proposition 1 and 2026 Theorem 4.1 are emitted only for integer
  (m\ge10).
- 2026 Theorem C.2 is emitted for integer (m\ge1).
- Every pairwise 2026 certificate additionally requires
  (d_{ij}^2>0) and
  (d_{ij}^2+(v_j-v_i)/m>0).
- Multiclass aggregation requires all ordered pairs (i,j), i\ne j; partial
  pair dictionaries are rejected rather than divided by the full class count.

Invalid curve points are serialized as JSON `null` together with a validity
record and reason. Plotters render a gap rather than a finite certificate.
