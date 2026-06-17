import os
import torch
from unsloth import FastLanguageModel
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from sentence_transformers.losses import ContrastiveLoss, MultipleNegativesRankingLoss
from datasets import Dataset
from sentence_transformers.models import Transformer, Pooling

import json
from patent_retrieval import dataset as patent_dataset, utils as utils
import sqlmodel as sqlm
from pyrootutils import setup_root
from pathlib import Path
import optuna



root = setup_root(__file__)
logger = utils.get_logger(__name__)
os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_DIR"] = str(root)
db_path = Path(os.environ["CLEF_IP_LOCATION"]) / "patents_v3.db"
train_topics_path=Path(os.environ["CLEF_IP_LOCATION"])/ "02_topics"/ "training-pac"/"clef-ip-2011_PACTraining"/"qrels.txt"
MAX_SEQ_LENGTH = 40960 -10  
MODEL_NAME = "Qwen/Qwen3-Embedding-4B" 
RUN_NAME = "optuna-qwen3-emb-4b-v3"  

engine = sqlm.create_engine(f"sqlite:///{db_path}")

def fetch_candidate_patent(patent_number: str) -> str:
    with sqlm.Session(engine) as session:
        patent = session.exec(
        sqlm.select(patent_dataset.Patent).where(patent_dataset.Patent.number == patent_number)
        ).first()
        if patent:
            return patent_dataset.extract_query_text(patent, search_columns=["title", "abstract", "claims"],kclaims=1)
        else:
            return ""
        
def fetch_topic_patent(patent_number: str) -> str:
    topic_file = next(
        train_topics_path.parent.glob(f"files/{patent_number}.xml")
    )
    topic_patent = patent_dataset.parse_patent([topic_file])[0]
    return patent_dataset.extract_query_text(topic_patent, search_columns=["title", "abstract", "claims"],kclaims=1)

def apply_qwen_template(x):
    # Retrieve raw texts
    q_text = fetch_topic_patent(x["anchor"])
    pos_text = fetch_candidate_patent(x["positive"])
    neg1_text = fetch_candidate_patent(x["negatives"][0])  
    neg2_text = fetch_candidate_patent(x["negatives"][1])

    # 2. Safety Check (Filter empty)
    if not (q_text and pos_text and neg1_text and neg2_text):
        return {"anchor": "", "positive": "", "negative1": "", "negative2": ""}

    # Anchor gets the "Query" prompt
    anchor = f"Instruct: Given a patent query, retrieve relevant patent documents.\nQuery: {q_text}\n"
    
    # Return 4 columns. The Trainer automatically maps strict column order:
    # Col 1 -> Anchor, Col 2 -> Positive, Col 3+ ->
    return {
        "anchor": anchor, 
        "positive": pos_text, 
        "negative1": neg1_text, 
        "negative2": neg2_text
    }

logger.info("Loading Data...")
raw_data = json.load(open("data/mnrl_finetuning_data.json", "r"))
dataset = Dataset.from_list(raw_data)

#Multiple Negatives Ranking Loss 
dataset = dataset.map(apply_qwen_template)
# Filter out failures (empty strings)
dataset = dataset.filter(
    lambda x: len(x["anchor"]) > 10 and 
              len(x["positive"]) > 10 and 
              len(x["negative1"]) > 10 and 
              len(x["negative2"]) > 10
) 


"""
#Contrastive loss
dataset = dataset.map(
    lambda x: {
        "sentence1": fetch_topic_patent(x["sentence1"]),
        "sentence2": fetch_candidate_patent(x["sentence2"])
    }
)
"""
# Split dataset for validation
dataset = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]


def objective(trial):

    # Suggest hyperparameters
    learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True)
    batch_size = trial.suggest_categorical("batch_size", [  4,8,12])
    num_epochs = trial.suggest_categorical("num_epochs", [ 5,8,12,15,20])
    lora_r = trial.suggest_categorical("lora_r", [8, 16, 32])
    lora_alpha = trial.suggest_categorical("lora_alpha", [8, 16, 32,64])
    warmup_steps = trial.suggest_categorical("warmup_steps", [100, 200])

    # Load model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=False,
    )

    # Enable LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj","gate_proj", "up_proj", "down_proj"],
        lora_alpha=lora_alpha,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )


   # --- 4. Initialize Sentence Transformer Wrapper ---
    # We define the pooling strategy MANUALLY for Qwen
    logger.info("Initializing SentenceTransformer Wrapper...")

    embedding_model = Transformer(model_name_or_path=MODEL_NAME, 
                                max_seq_length=MAX_SEQ_LENGTH)
    # Re-assign the Unsloth model to the Transformer wrapper
    embedding_model.auto_model = model
    embedding_model.tokenizer = tokenizer

    # Qwen uses EOS token pooling (Last token), NOT Mean pooling
    pooling_model = Pooling(
        embedding_model.get_word_embedding_dimension(),
        pooling_mode="lasttoken", # <--- CRITICAL  for Qwen
        pooling_mode_mean_tokens=False,
        pooling_mode_cls_token=False,
        pooling_mode_max_tokens=False,
    )

    st_model = SentenceTransformer(modules=[embedding_model, pooling_model])

    loss = MultipleNegativesRankingLoss(st_model)
    output_dir : Path = root / "finetuning"/ "runs" / RUN_NAME / f"trial_{trial.number}"
    args = SentenceTransformerTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,  # Very small batch size for 4B/7B model + Long Context
        gradient_accumulation_steps=4,  # Simulate larger batch (Effective BS = 16)
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        bf16=torch.cuda.is_bf16_supported(),
        #fp16=not torch.cuda.is_bf16_supported(),
        logging_steps=10,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        optim="adamw_torch", # Use 8-bit optimizer to save more VRAM
    )

    trainer = SentenceTransformerTrainer(
        model=st_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=loss,
        args=args,
    )

    # Train and get validation loss
    trainer.train()
    eval_result = trainer.evaluate()
    
    # Clean up to free memory
    del model, st_model, trainer
    torch.cuda.empty_cache()
    
    return eval_result["eval_loss"]


# Run Optuna study
study = optuna.create_study(direction="minimize", study_name="patent_embedding")
study.optimize(objective, n_trials=100,show_progress_bar=True)

# Print results
print("Best trial:")
print(f"  Value: {study.best_trial.value}")
print("  Params: ")
for key, value in study.best_trial.params.items():
    print(f"    {key}: {value}")

# Train final model with best params
best_params = study.best_trial.params
logger.info(f"Training final model with best params: {best_params}")

# Save trial summary
study.trials_dataframe().to_csv(root / "finetuning" / "runs" / RUN_NAME / "trial_summary.csv", index=False)
logger.info(f"Saved trial summary to {root / 'finetuning' / 'runs' / RUN_NAME / 'trial_summary.csv'}")

# Save best parameters
with open(root / "finetuning" / "runs" / RUN_NAME / "best_params.json", "w") as f:
    json.dump(best_params, f, indent=2)
logger.info(f"Saved best parameters to {root / 'finetuning' / 'runs' / RUN_NAME / 'best_params.json'}")
# ... (repeat training code with best_params for final model)