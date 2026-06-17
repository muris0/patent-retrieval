# Finetuning

This folder contains the embedding-model fine-tuning workflow used for patent retrieval.
It covers data preparation, model adaptation, training, optional hyperparameter optimization,
and evaluation of the resulting retriever. The training setup supports both Standard
Contrastive Loss and Multiple Negatives Ranking Loss (MNRL); the current training scripts
instantiate MNRL by default.

## Workflow

1. Run retrieval to generate candidates
	- Use `01_ft_retriever.py` to retrieve candidate documents for the training topics.
	- Output is saved as `data/results.csv`.
2. Prepare finetuning data
	- Use `02_data_preparation.ipynb` to convert the retrieval output into training triplets.
	- Output is saved as `data/mnrl_finetuning_data.json`.
3. Finetune the embedding model
    - two vairants
        - Native Unsloth path: `03_finetune.py`
        - HF-based path: `03_finetune_hf-based.py`, in case the model is not supported by unsloth
    -Set key hyperparameters

5. Hyperparameters Optimization with optuna
	- Similar to the privious step
        - `04_finetune_optuna.py` (native path)
        - `04_finetune_optuna_hf-based.py` (HF-based path)


The training code supports two contrastive objectives:

- `MultipleNegativesRankingLoss` for in-batch negative sampling.
- `ContrastiveLoss` for setups that use explicit negative examples.

## Script Selection

- Use `03_finetune.py` when the model is supported natively by Unsloth's SentenceTransformer integration.
- Use `03_finetune_hf-based.py` for models that are not supported natively by that integration.
- The two `04_finetune_optuna*.py` scripts are HPO (Hyperparameter Optimization) variants of the same two training paths.

## Why Unsloth

Unsloth is used because it makes this training setup more practical:

- Lower memory usage through LoRA and Unsloth gradient checkpointing
- Faster fine-tuning throughput

## Outputs

Training outputs are saved under:

- `finetuning/runs/{RUN_NAME}_{epochs}epochs/`

Intermediate data artifacts used by the workflow are saved under:

- `data/results.csv`
- `data/mnrl_finetuning_data.json`

