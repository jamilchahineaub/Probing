# Run artifacts

Each subdirectory is immutable and identified by experiment, UTC timestamp,
seed, and a collision-resistant suffix. The manifest is the provenance entry
point. Reproduction creates a new run; old runs are never overwritten.

Current Milestone A runs save raw time histories as both compressed CSV
(`raw_timeseries.csv.gz`) and compressed NumPy (`raw_timeseries.npz`) data.

EXP-0002 saves every Monte Carlo trial in compressed CSV/NPZ form, aggregate
distribution and GO-candidate tables as CSV, and a smaller representative
time-series subset in compressed CSV/NPZ form. The manifest links all files.

EXP-0003 separately saves timing trials, timing-profile trials, held-out
identification trials, aggregate distributions, timing thresholds, parameter
classifications, representative time series, stage decision, and nine PNG/PDF
diagnostic figure families. Its manifest also records EXP-0001/0002 artifact
fingerprints before and after execution.

EXP-0004 saves every executed probe stage and candidate-selection evaluation in
compressed CSV/NPZ form, stage and final strategy summaries, frequency-
information diagnostics, representative sequential time series, the kill/gate
decision, and nine PNG/PDF figure families. Its manifest records EXP-0001/0002/
0003 fingerprints before and after execution.

EXP-0005 saves all held-out predictor rows in compressed CSV/NPZ form,
classification and quantitative-response summaries, risk-class and dynamics
breakdowns, mass-error/decision comparisons, representative causal probe time
series, false-safe and safety audits, the kill/gate decision, and eight PNG/PDF
figure families. Its manifest records EXP-0001 through EXP-0004 fingerprints
before and after execution.

EXP-0006 saves all fixed-duration validation predictions and causal early-stop
rows in compressed CSV/NPZ form, duration/feature/quantitative summaries,
ring-down feature importance, the locked EXP-0005 false-safe audit,
representative passive time series, acceptance and Stage decisions, and nine
PNG/PDF figure families. Its manifest records EXP-0001 through EXP-0005
fingerprints before and after execution.
