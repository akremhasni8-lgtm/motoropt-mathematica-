# Engine Multi-Objective Bayesian Optimization (Wolfram Language)

Native Wolfram Language implementation of multi-objective Bayesian
optimization for internal-combustion-engine calibration, using
`Predict[...]` Gaussian Process oracle models as digital-twin stand-ins for
Torque, exhaust temperature (T4), and NOx emissions.

This is the Mathematica counterpart of [`motoropt-py-`](../motoropt-py-),
which solves the same problem with Ax/BoTorch in Python.

## Approach

Rather than running the physical/simulated engine for every candidate
calibration, the notebook trains **oracle Gaussian Process models**
(combined Squared-Exponential + Rational-Quadratic covariance) on a random
subsample of the DoE dataset, then runs a **batch-synchronous Bayesian
optimization loop** against those oracles to search for the Pareto front
between torque, exhaust temperature, and NOx.

### Pipeline

1. **Load data** — `engine_gpr_dataset.xlsx`, 19 columns: 10 calibratable
   actuator parameters (`APP_r`, `EGR_r`, injection curve parameters, throttle,
   etc.), 5 fixed operating-point/context variables (ambient temp, coolant
   temp, battery voltage, engine speed, rail pressure), and 3 targets
   (`Torque_Nm`, `T4_degC`, `NOx_ppm`). Rows 2–5001 = DoE training set, rows
   5002–6801 = WHTC validation set.
2. **Fit bounds & fixed context** — design-space bounds from the actuator
   columns; context fixed at the median operating point.
3. **Train oracle GP models** — `Predict[...]` with
   `Method -> {"GaussianProcess", "CovarianceType" -> "SquaredExponential" + "RationalQuadratic"}`
   on `nOracleTrain = 600` sampled points, used as ground-truth stand-ins for
   Torque, T4, and NOx.
4. **Bayesian optimization loop** — warm-start sample of `nInitTrain = 60`
   points, `nIterations = 8` BO iterations, `batchSize = 4` candidates per
   iteration (batch-synchronous), acquisition maximized over a random-search
   candidate pool of `nCandidatesAcq = 1500` points per iteration.
5. **Feasibility constraint** — candidate points are feasible when T4 is
   below the `t4Percentile = 0.75` threshold of the observed distribution.
6. **Output** — Pareto front of Torque vs NOx under the T4 feasibility
   constraint (`pareto_front_engine.png` equivalent), random seed `2026`.

## Files

```
MultiObjective_BO_Engine_ParetoFront_1.wl   # main BO script
mutiobj opt.nb                               # exploratory notebook
mutiobj opt (mathematica).pdf                # exported report/printout
engine_gpr_dataset.xlsx                      # shared dataset (see motoropt-py-)
config.png / pareto_front_engine.png         # configuration & result plots
```

## Requirements

- Wolfram Mathematica 14.1+ (uses `Predict`, `NearestFunction`, built-in GP
  covariance kernels — no external packages required)

## Related work

Companion Python implementation: [`motoropt-py-`](../motoropt-py-) (Ax +
BoTorch). Both target the same underlying research question — see also the
Bayesian Optimization chapter in [`PFA2`](../PFA2) for the theoretical
background (GP covariance functions, acquisition functions) used here.

## Author

Akrem Hasni — Mechanical Engineering, ENSIT.
