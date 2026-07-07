# Federated Learning Algorithm Comparison (CIFAR-10 Non-IID)

## Project Purpose
Compare 5 Federated Learning aggregation algorithms across 5 types of non-IID data distributions using CIFAR-10 dataset.

**Algorithms**: FedAvg, FedProx, FedNova, SCAFFOLD, MOON  
**Non-IID Types**: IID, Label Skew, Feature Skew, Quantity Skew, Concept Shift  
**Grid Size**: 5 × 5 = 25 experiment combinations

## Notebook Structure

### Core Sections (Execution Order)
1. **Cell 2**: Imports & device setup
2. **Cell 3** (CONFIG): All experiment parameters in one place — **START HERE for parameter tuning**
3. **Cell 4**: Download CIFAR-10 and visualize samples
4. **Cell 5-6**: Non-IID partition strategies (IID, Label Skew, Quantity Skew, Feature Skew, Concept Shift)
5. **Cell 7-10**: Model architecture (SimpleCNN with GroupNorm) and training implementations (5 algorithms)
6. **Cell 11**: Quick single-run test with current CONFIG
7. **Cell 12-13**: Full grid experiment runner (25 combinations) with result persistence
8. **Cell 14**: Visualization & analysis of all results

### Key Configuration (Cell 3)
All parameters in `CONFIG` dict:
- **FL Settings**: `num_clients`, `num_rounds`, `local_epochs`, `lr`, `batch_size`, etc.
- **Data**: `train_subset_size`, `test_subset_size` (use smaller values for quick tests)
- **Non-IID**: `partition` type, `label_skew_alpha`, `quantity_skew_alpha` (smaller α = stronger skew)
- **Algorithm**: `algorithm` name, algorithm-specific params (`fedprox_mu`, `scaffold_server_lr`, `moon_mu`)
- **Reproducibility**: `seed` controls all RNG (train/test splits, initialization, stochasticity)

## Common Tasks

### Run Quick Test (1 config)
Modify `CONFIG["algorithm"]` and `CONFIG["partition"]`, then run Cell 11 to verify setup before full grid.

### Run Full Grid (25 combinations)
Run Cell 12 to execute all 25 experiments. Results auto-saved to `fl_experiment_results.json` after each combo → resume-safe if interrupted.

### Tune for Specific Non-IID Type
1. Set `CONFIG["partition"]` to desired type (e.g., "label_skew")
2. Modify skew intensity via alpha parameters (smaller = stronger skew)
3. Run Cell 11 to test before grid
4. Use Cell 14 visualization to compare algorithms for that partition type

### Analyze Results
Cell 14 generates:
- **Line plots**: Accuracy evolution per algorithm per non-IID type
- **Heatmap**: Final accuracy table (algorithms × non-IID types)
- **CSV**: Saved as `fl_summary_table.csv`

Raw data in `fl_experiment_results.json`: `{"{partition}__{algorithm}": [{"round": r, "test_acc": a, "test_loss": l}, ...], ...}`

## Critical Implementation Details

### Partition Functions (Cell 5)
- **`partition_iid`**: Random uniform split
- **`partition_label_skew`**: Dirichlet(alpha) per-class allocation
- **`partition_quantity_skew`**: Dirichlet(alpha) on total samples, preserves class ratio
- **`partition_group_iid`**: Base for Feature Skew / Concept Shift (with client groups)

### Algorithm Training (Cell 8-9)
Each algorithm has dedicated `local_train_XXX()` function:
- **FedAvg**: Standard SGD + CrossEntropy
- **FedProx**: FedAvg + proximal regularization term
- **FedNova**: FedAvg with tau (step count) normalization at aggregation
- **SCAFFOLD**: SGD with control-variate gradient correction (uses SGD, not momentum)
- **MOON**: SGD + contrastive loss pulling local features toward global, pushing away from previous local

### Aggregation (Cell 9)
- **FedAvg/FedProx/MOON**: `weighted_avg_state_dicts()` — weighted average by client sample counts
- **FedNova**: `fednova_aggregate()` — delta normalized by local step count before averaging
- **SCAFFOLD**: `scaffold_aggregate()` — sum of deltas averaged uniformly across ALL clients (not selected clients)

### Test Set
Uses `DEFAULT_TRANSFORM` (uniform preprocessing) applied to original test data labels. 
⚠️ Note: Not specialized per-partition; represents global generalization, not per-partition performance.

## Pitfalls & Best Practices

| Issue | Solution |
|-------|----------|
| Grid takes 40-60 min on T4 GPU | Use smaller `train_subset_size` / `num_rounds` for quick testing |
| Results lost if Colab disconnects | Mount Google Drive, set `RESULTS_FILE` to Drive path |
| Single seed only → high variance | Current design; re-run with different `seed` values externally |
| Can't detect which algo struggles per non-IID | Parse `fl_experiment_results.json` and compute per-partition rankings |
| Unfair algo comparison if configs aren't tuned | Each algorithm has different optimal hyperparams (`fedprox_mu`, `moon_mu`, etc.); tune per-algorithm separately |

## Files Generated
- `fl_experiment_results.json`: Full history (accuracy/loss per round per combination)
- `fl_summary_table.csv`: Summary table of final accuracies

## Recommended Enhancements
- Add per-seed averaging (run grid multiple times with different seeds, compute mean ± std)
- Add convergence speed metrics (round-to-reach-target-accuracy)
- Add per-client accuracy distribution analysis (especially for Feature Skew / Concept Shift)
- Add ablation studies on alpha parameters to show skew intensity effects
