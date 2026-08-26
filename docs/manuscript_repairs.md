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
| 15 | “general few-shot guarantee in (??)” | Luthra et al. (2025), Proposition 1; see the normalization repair below |
| 17 | “few-shot NCC guarantees from Thm. ??” | Theorem 4.5 |

Use labeled `\eqref{...}` / `\Cref{...}` references rather than hard-coded
numbers for internal references, then build twice and fail the submission check
if the PDF text still contains `??`. The page-15 few-shot source is an external
citation, not an internal theorem reference.

## 6. Repair the Appendix A few-shot bound

Equation (9) cannot be obtained from Theorem 4.5. Theorem 4.5 requires a
centered, whitened representation, whereas (F^\star_\gamma) has
mode-dependent scales. Rewhitening it would recover the selected subspace and
the capture (B_r), eliminating the claimed (\gamma)-dependent expression.

The algebraic form of Equation (9) instead comes from the fixed-(a=16)
corollary in Luthra et al. (2025), Proposition 1. That paper's declared metrics
are ordered single-class quantities. For the balanced binary case, write the
draft's unhalved symmetric quantities as

\[
A_\gamma=\frac{C_\gamma}{B_\gamma^2}-1,
\qquad
D_\gamma=\frac{S_\gamma}{B_\gamma}-1.
\]

Then the ordered/original-half metrics are

\[
\widetilde V_{\rm ord}=\frac{A_\gamma}{4},
\qquad
V_{\rm ord}=\frac{D_\gamma}{4},
\qquad
V^s_{\rm ord}\leq\sqrt{V_{\rm ord}}
=\frac12\sqrt{D_\gamma},
\]

where the last step is Jensen's inequality if only the aggregate variance is
retained. Consequently, a provenance-correct three-scalar corollary for
(m\ge10) is

\[
\operatorname{err}^{\rm NCC}_m(F^\star_\gamma)
\le
2A_\gamma
+\frac{4}{\sqrt m}\sqrt{D_\gamma}
+\left(\frac{2}{\sqrt m}+\frac1m\right)D_\gamma.
\]

Alternatively, retain the two class-specific variances and substitute the
exact ordered (V^s_{\rm ord}) rather than applying Jensen. Do not describe the
current Equation (9) as a direct substitution into Theorem 4.5. If a looser
unhalved conversion is intentionally kept, label it as a conservative
conversion and show every factor-of-two step explicitly.

## 7. Repair the Theorem 4.5 zero-capture case and reporting language

The proof defines (u=w/\lVert w\rVert), which is undefined when (B(F)=0).
Begin the proof with:

> If (B(F)=0), the result follows from
> (\operatorname{err}^{\rm NCC}_m(F)\le1\le2+r/m). Hence assume
> (B(F)>0) below.

State the equality with the directional-CDNV expression for (B(F)>0), with its
continuous extended-value interpretation at (B(F)=0).

The displayed theorem right-hand side must be retained in tables before any
probability clipping. If a figure plots (\min\{1,\mathrm{RHS}\}), label that as
display-only clipping. On a balanced binary task, distinguish:

- (\mathrm{RHS}\ge1): probability-vacuous;
- (1/2\le\mathrm{RHS}<1): nontrivial relative to the probability ceiling but
  not informative relative to chance;
- (\mathrm{RHS}<1/2): guarantees error below the chance level.

## 8. Submission checks

After the source becomes available:

1. build the manuscript twice;
2. extract text and assert there is no `??`;
3. assert the CDNV definition includes the normalization sentence;
4. assert the spectral setting includes “complete orthonormal eigenbasis”;
5. assert the Appendix B optimizer contains (C_{12}^{\top});
6. assert Appendix A cites Luthra et al. Proposition 1, states (m\ge10), and
   uses an explicit CDNV conversion;
7. assert Theorem 4.5 handles (B(F)=0) before defining (u);
8. replace the superseded PDF and record its source commit and SHA-256.
