# Exact ACT training metrics

`train_with_metrics.py` runs the installed LeRobot trainer and attaches a standard logging handler after logging initialization. It reads the training `MetricsTracker` before terminal formatting and meter reset. The optimizer, sampler, losses and shared environment source remain unchanged.

Use a separate dataset copy and a new output directory for each experiment. Save the actual configuration and dataset manifest before training. Do not reuse the historical run directory.

```bash
python ACT-1-Pick/train_with_metrics.py --metrics-dir <RUN>/metrics --config_path=<CONFIG.json>
```

The launcher forwards the remaining arguments to the official trainer. On machines using a shared GPU queue, submit this command through that queue. The launcher itself does not submit jobs, upload data or operate a robot. Upload and W&B behavior remain controlled by the training configuration.

The metric directory must be new. CSV and JSONL preserve exact steps and original floating-point interval means for total loss, action L1, unweighted KL, gradient norm, learning rate, update time and loading time. Records flush immediately. Missing required fields, duplicate steps, nonfinite metrics or write errors stop the run with an error.

Test the complete chain with a separate 400-step probe and `log_freq=200` before starting the formal run from step zero. Do not resume the probe to obtain the formal model.

After the official trainer exits successfully, verify the saved records and checkpoints:

```bash
python ACT-1-Pick/verify_training_run.py --config <CONFIG.json> --metrics-dir <RUN>/metrics --output <RUN>/integrity.json
python ACT-1-Pick/plot_training_curve.py --input <RUN>/metrics/metrics.csv --output <RUN>/figures/training.png
```

The verifier checks exact expected steps, CSV/JSONL agreement, finite values, the ACT relation `loss = L1 + kl_weight × KL` within floating-point tolerance, checkpoint steps, saved configuration and model hashes. Plotting separates the three metrics and leaves gaps between discontinuous or resumed fragments. The saved CSV can regenerate the PNG without loading a policy.

For a 100,000-step run recorded every 200 steps and saved every 20,000 steps, expect 500 metric records and five checkpoints. A successful verification establishes training-record integrity. It does not establish validation accuracy, generalization or physical robot acceptance.
