# Manuscript repair patch

The workspace contains two compiled copies, `dirCDNV_is_low.pdf` and
`dirCDNV_is_low-17 (1).pdf`; text extraction confirms that both contain the
defects below. No manuscript `.tex`, `.bib`, or Overleaf source exists in any
local branch, worktree, or checked archive (the only discovered TeX file is an
unrelated meeting document). This file is the source-ready repair
specification. Both compiled PDFs must remain marked superseded until these
edits are applied to the actual source and rebuilt.

Audited PDF SHA-256 values:

- `dirCDNV_is_low.pdf`:
  `2707685fbd9dd23fe80d936552c6df7a452bf41e69c21b071e6981b8644e27b8`
- `dirCDNV_is_low-17 (1).pdf`:
  `7aadfbf88438583cbf5054ab26c2eee64c43e690127796463af87b1407ea19b9`

## 1. Freeze the CDNV convention

Replace any claim that the paper uses the original CDNV normalization with:

> We use the unhalved symmetric convention
> \[
> V_F(i,j)=\frac{\operatorname{tr}\Sigma_i+
> \operatorname{tr}\Sigma_j}{\|\mu_i-\mu_j\|^2},\qquad
> \widetilde V_F(i,j)=\frac{u_{ij}^{\top}(\Sigma_i+\Sigma_j)u_{ij}}
> {\|\mu_i-\mu_j\|^2}.
> \]
> Galanti et al. (2022) define CDNV with an additional factor (1/2).
> Thus our (V_F) and symmetrized (\widetilde V_F) are twice their
> half-normalized counterparts.

Under this convention Proposition 4.1 remains

\[
V_F=\frac{r-B}{2B},\qquad \widetilde V_F=\frac{1-B}{2B}.
\]

If the original half-normalization is used in a comparison, write instead
((r-B)/(4B)) and ((1-B)/(4B)). The implementation contract is in
`docs/cdnv_conventions.md`.

## 2. Strengthen the spectral assumptions before Proposition 3.1

The proof expands arbitrary functions in a complete eigenbasis, which does not
follow from the phrase “orthonormal eigensystem.” Insert the following
assumption:

> **Spectral setting.** Let (X^{(1)},X^{(2)}) be exchangeable and
> conditionally i.i.d. given the latent instance (Z), with common marginal
> (P_X). Define
> \[
> (Tf)(x)=\mathbb E[f(X^{(2)})\mid X^{(1)}=x].
> \]
> We work either in a finite-dimensional (T)-invariant hypothesis space, or
> assume that (T) is compact on the relevant closed subspace of
> (L^2(P_X)). Then (T) is a self-adjoint positive-semidefinite contraction:
> \[
> \langle g,Tf\rangle
> =\mathbb E[\mathbb E(g(X)\mid Z)\mathbb E(f(X)\mid Z)]
> =\langle Tg,f\rangle,
> \qquad \langle f,Tf\rangle\ge0.
> \]
> The spectral theorem therefore supplies a complete orthonormal eigenbasis
> (including the kernel when needed), which is the basis used below.

State (\lambda_r>0) wherever the top-(r) result needs a positive retained
mode. A boundary eigengap (\lambda_r>\lambda_{r+1}) is needed only for
uniqueness/stability of the selected top-(r) subspace, not for the optimal
value. The experiment metadata now serializes the full empirical spectrum,
(\lambda_r), and the boundary eigengap.

## 3. Correct the Appendix B regression orientation

For the convention (C_{12}=\mathbb E[F_1F_2^\top]) and loss
(\mathbb E\|F_2-WF_1\|^2), replace the optimizer line by

\[
W^\star C_{11}=C_{21}=C_{12}^{\top},
\qquad
W^\star=C_{21}=C_{12}^{\top}\quad(C_{11}=I).
\]

Then explicitly invoke exchangeability to state (C_{12}=C_{21}), if the
subsequent argument wants to write (W^\star=C_{12}).

## 4. Correct hyperrectangle terminology

Every predicted coordinate is (y_t\sqrt{B_t}). Therefore:

- (\sqrt{B_t}) is the **half-side** along task (t);
- (2\sqrt{B_t}) is the full edge length;
- (B_t) is captured energy, not a side length.

Replace “side length (\sqrt{B_t})” and “side length (B_t)” accordingly on
pages 2 and 7 of the compiled draft.

## 5. Resolve the four broken references

The compiled draft contains these unresolved references:

| PDF page | Broken text | Required target |
|---:|---|---|
| 14 | operator “introduced in (??)” | the earlier displayed definition of (T) |
| 15 | means/covariances “defined in (??)” | the display defining (\mu_\pm,\Sigma_\pm) |
| 15 | “general few-shot guarantee in (??)” | Theorem 4.5 |
| 17 | “few-shot NCC guarantees from Thm. ??” | Theorem 4.5 |

Use labeled `\eqref{...}` / `\Cref{...}` references rather than hard-coded
numbers, then build twice and fail the submission check if the PDF text still
contains `??`.

## 6. Submission checks

After the source becomes available:

1. build the manuscript twice;
2. extract text and assert there is no `??`;
3. assert the CDNV definition includes the normalization sentence;
4. assert the spectral setting includes “complete orthonormal eigenbasis”;
5. assert the Appendix B optimizer contains (C_{12}^{\top});
6. replace the superseded PDF and record its source commit and SHA-256.
