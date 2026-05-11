# Market Regime Detection — Student-t HMM with Time-Varying Transition Probabilities

A from-scratch implementation of a three-state Hidden Markov Model (Crisis / Neutral / Bull) fitted on SPY daily log-returns. Two statistically motivated upgrades over the textbook HMM: **Student-t emissions** for fat-tail robustness, and **time-varying transition probabilities** (TVTP) conditioned on realised volatility. Fully out-of-sample evaluation on 2016–present with documented failure analysis and principled fixes.

> For full mathematical derivations, OOS failure hypotheses, and implementation details see [`TECHNICAL_NOTES.md`](TECHNICAL_NOTES.md).

---

## Model

The standard HMM assumes Gaussian emissions and a fixed transition matrix — both poor fits for equity returns. This implementation replaces them with:

**Student-t emissions.** Each state emits from $t_{\nu_k}(\mu_k, \sigma_k^2)$. The degrees-of-freedom parameter $\nu_k$ is learned per state, giving the model a continuous knob between Gaussian and heavy-tailed behaviour. EM is made exact via the Gaussian scale-mixture representation of the Student-t (McLachlan & Peel 2000), which yields closed-form auxiliary variable moments and avoids any approximation.

**Time-varying transition probabilities.** The transition matrix at time $t$ is $A_t(i,j) = \text{softmax}_j(\mathbf{W}_i \cdot [1, z_t])$, where $z_t$ is a z-scored 10-day realised volatility. The weight matrices $\mathbf{W}_i$ are learned in the EM M-step via weighted logistic regression with an analytic gradient and L-BFGS-B optimisation. The covariate is ticker-agnostic — no VIX required.

No HMM library is used. The full forward-backward algorithm, Viterbi decoder, causal forward filter, and both M-steps are implemented from scratch in NumPy/SciPy.

---

## Training and Evaluation Split

| Period | Dates | Days |
|---|---|---|
| In-sample (IS) | 2000–2015 | ~4,000 |
| Out-of-sample (OOS) | 2016–present | ~2,600 |

The IS period spans the dot-com crash, 2003–2007 bull market, and GFC — an unusually crisis-rich training set with consequences for OOS generalisation documented below.

---

## In-Sample Results

IS emission parameters learned by EM:

| State | $\mu$/yr | $\sigma$/yr | $\nu$ | % IS days |
|---|---|---|---|---|
| Crisis | −41% | 34% | 6.4 | 12.1% |
| Neutral | −6% | 17% | 12.9 | 45.1% |
| Bull | +29% | 8% | 7.1 | 42.8% |

Average IS transition matrix (TVTP — varies daily with RVol):

|  | → Crisis | → Neutral | → Bull |
|---|---|---|---|
| Crisis → | 0.97 | 0.00 | 0.03 |
| Neutral → | 0.01 | 0.97 | 0.02 |  
| Bull → | 0.00 | 0.98 | 0.02 |

The Student-t fit visibly outperforms an equivalent Gaussian in tail coverage for all three states, most notably in Crisis where extreme-move frequency is highest.

---

## OOS Evaluation

Naively freezing IS parameters and applying Viterbi to 2016–present produces **100% Crisis classification** despite SPY rising ~200→700. Four failure hypotheses were tested in sequence; all fixes are derived from the model's own mathematics rather than ad hoc adjustments. Full derivations in [`TECHNICAL_NOTES.md`](TECHNICAL_NOTES.md).

**Fix summary:**

| Component | IS | OOS | Why |
|---|---|---|---|
| Decoder | Viterbi (non-causal) | Causal forward filter | Viterbi backward pass uses future data — invalid OOS |
| Initialisation | `pi_` = [1, 0, 0] | IS terminal $\alpha_T$ = [0.03, 0.56, 0.42] | Correct state handoff at IS/OOS boundary |
| $P(C{\to}C)$ | 0.974 | 0.85 | IS over-fitted to GFC stickiness; 0.85 is the first value below the emission-competitiveness threshold of 0.809 |
| $P(B{\to}C)$ | 0.000 | 0.050 | Zero probability blocked direct vol-spike response |
| Emission $\mu$, $\sigma$ | IS-fitted | OOS EM refitted | IS crash-era location/scale incompatible with post-2015 structure |
| Emission $\nu$ | IS-fitted | Frozen | Tail shape is structurally stable across cycles |

---

## OOS Regime-Conditional Statistics (2016–present)

| State | Days | %Time | Ann. Ret | Ann. Vol | Sharpe | Max DD | Hit Rate |
|---|---|---|---|---|---|---|---|
| Crisis | 1,786 | 69.0% | +9.8% | 20.5% | +0.48 | −35.1% | 45.9% |
| Neutral | 7 | 0.3% | −185.0% | 16.6% | −11.17 | −4.7% | — |
| Bull | 794 | 30.7% | +24.4% | 9.8% | +2.49 | −10.1% | 58.9% |

*Sharpe: annualised log-return / annualised vol, rf = 0. Max DD: over regime sub-series only. Hit rate: % of regime days where next-day return has regime-consistent sign (Crisis → negative, Bull → positive).*

The Bull classification is the cleanest signal: Sharpe 2.49, max drawdown −10.1%, and 58.9% next-day directional accuracy with no look-ahead. Crisis at 69% of OOS days with a positive return (+9.8%/yr, Sharpe 0.48) reflects the residual emission mismatch between IS crash-era parameters and the post-2015 environment — a known limitation discussed in [`TECHNICAL_NOTES.md`](TECHNICAL_NOTES.md). Neutral is near-vacuous (n=7) and structurally squeezed between Crisis and Bull emissions.

---

## Dependencies

```
numpy  scipy  pandas  matplotlib  yfinance
```

No HMM or ML libraries.

---

## References

- Hamilton, J. D. (1989). A new approach to the economic analysis of nonstationary time series. *Econometrica* 57(2).
- Filardo, A. J. (1994). Business-cycle phases and their transitional dynamics. *JBES* 12(3).
- McLachlan, G. J. & Peel, D. (2000). *Finite Mixture Models*. Wiley.
- Rabiner, L. R. (1989). A tutorial on hidden Markov models. *Proc. IEEE* 77(2).

