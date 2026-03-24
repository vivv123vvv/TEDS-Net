# AGENT.md

This file defines repository-wide rules for future agents and collaborators.

## 1. Project Scope

- This repository is centered on TEDS-Net: segmentation is produced by deforming an explicit prior shape, not by predicting a mask directly from the image.
- The current research direction is to keep the "prior shape plus topology-aware deformation" paradigm while replacing the original TEDS-Net topology-preserving integration block with the LC-ResNet module described in the R2Net paper.
- Unless the user explicitly asks for it, do not turn this project into a standard direct segmentation network.

## 2. Paper References

- TEDS-Net paper:
  `docs/papers/teds-net-topology-preservation.pdf`
- R2Net / LC-ResNet paper:
  `docs/papers/r2net-lc-resnet-diffeomorphic-registration.pdf`

When making structural changes, use these two papers as the primary references. If the implementation intentionally differs from either paper, record the reason in code comments, commit notes, or supporting docs.

## 3. Current Code Map

- `scripts/train_runner.py`: training entry point
- `scripts/trainer.py`: training, validation, and evaluation flow
- `scripts/network/TEDS_Net.py`: top-level TEDS-Net assembly
- `scripts/network/UNet.py`: encoder, bottleneck, decoder, and base network blocks
- `scripts/network/utils_teds.py`: flow generation, integration, warping, and spatial transform logic
- `scripts/dataloaders/`: MNIST and ACDC loaders
- `scripts/parameters/`: dataset and network hyperparameters
- `scripts/utils/losses.py`: Dice loss and deformation regularization losses

If new preprocessing, training, evaluation, or visualization scripts are added, keep responsibilities separated. Do not pack unrelated pipeline stages into one file.

## 4. Non-Negotiable Research Constraints

- Preserve the core pattern: image input plus explicit prior shape input plus deformation of that prior.
- If LC-ResNet, diffeomorphic, or integrator code is modified, keep these interface assumptions stable:
  - displacement-field channel count matches spatial dimensionality
  - `WholeDiffeoUnit` outputs remain compatible with `WarpPriorShape` and the spatial transformer
  - upsampled flow size matches the transformer grid exactly
  - coordinate ordering, axis permutation, and `align_corners` behavior are not changed casually
- Do not silently remove MNIST smoke-test support. It is the smallest regression baseline in this repo.
- Do not duplicate hard-coded dataset paths, patient splits, or label semantics across files. Prefer a single parameter or configuration source.

## 5. Rules For Experimental Changes

- Keep algorithm experiments separate from engineering fixes whenever practical.
- If a change touches any of the following, state the experiment goal clearly:
  - prior-shape generation
  - integrator or deformation layer
  - loss weights
  - data slicing, resizing, filtering, or label handling
  - decoder depth, feature-map count, or branch structure
- Do not silently change ACDC label semantics. The default assumption is myocardium-only segmentation with label `2`. If multi-class behavior is introduced, document it explicitly.

## 6. Validation Expectations

- Any change to network structure or deformation logic should complete at least two of the following when possible:
  - import or build smoke test
  - single-batch forward-pass check
  - tensor-shape verification
  - CPU/GPU compatibility check
  - backward-pass or loss propagation check
  - Jacobian, folding, or topology-related check
- If training or evaluation cannot be run, say exactly what was not run, why it was not run, and what risk remains.
- If data preprocessing or dataset splitting changes, document train, validation, and test sources and note any leakage risk.

## 7. Documentation And Traceability

- When adding or replacing a paper-derived module, update `README.md` or supporting docs to explain:
  - what changed
  - why it changed
  - where the entry points are
  - how to validate it
- Store papers, reports, and experiment notes under `docs/`.
- If a temporary script becomes part of the workflow, promote it into a clearly named maintained script instead of leaving it as an ad hoc root-level file.

## 8. Branch And Collaboration Rules

- Keep `main` readable and explainable. Do not mix multiple unrelated research hypotheses into one change without clear justification.
- Prefer separate branches for new experiments. If Codex creates a branch, use the `codex/` prefix.
- Do not overwrite or revert user work unless the user explicitly requests it.

## 9. Default Agent Behavior

- Protect the research intent before optimizing for superficial "it runs" behavior.
- When paper behavior and repository behavior differ, identify the difference before editing code.
- In explanations, distinguish clearly between:
  - paper intent
  - current repository behavior
  - behavior introduced by the latest change

