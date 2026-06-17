import os
import torch
import json
import sqlmodel as sqlm
from pathlib import Path
from datasets import Dataset
from unsloth import FastLanguageModel, FastModel
from sentence_transformers import (
    SentenceTransformer, 
    SentenceTransformerTrainer, 
    SentenceTransformerTrainingArguments
)
import rich.syntax

import hydralette as hl
from sentence_transformers.models import Transformer, Pooling
from sentence_transformers.losses import MultipleNegativesRankingLoss, ContrastiveLoss
from patent_retrieval import dataset as patent_dataset, utils as utils
from pyrootutils import setup_root

# --- 0. Setup & Config ---
os.environ["WANDB_DISABLED"] = "true"
root = setup_root(__file__)
logger = utils.get_logger(__name__)

# CONFIGURATION
# Note: 32k context is massive. If you get OOM (Out of Memory), reduce this to 8192 or 4096.
MAX_SEQ_LENGTH = 40950  
MODEL_NAME = "Qwen/Qwen3-Embedding-4B" # Note: 'Qwen3' is not standard yet. Replaced with valid Qwen2.5 ID. Use your specific path if local.
RUN_NAME = "patQWEN3-emb-8b-lora-v1"

def get_run_dir(cfg: hl.Config) -> Path:

    out_dir: Path = root / "finetuning"/ "runs" / (f"{cfg.run_name}_{cfg.epochs}epochs")
    
    if out_dir.exists():
        out_dir = (
            out_dir.parent
            / (f"{cfg.run_name}")
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir

cfg = hl.Config(
    run_name=RUN_NAME,
    model_name=MODEL_NAME,
    epochs=5,
    learning_rate=1e-5,
    warmup_steps=100,
    max_seq_length=40950,
    optimizer="adamw_torch",
    lora_alpha=32,
    lora_r=16,
    lora_dropout=0.0,
    pooling_mode="lasttoken",
    loss_type="MultipleNegativesRankingLoss",
    lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj","gate_proj", "up_proj", "down_proj"],
    output_dir=hl.Field(reference=get_run_dir, type=Path),
    batch_size=8,

)
cfg.apply()
rich.print(rich.syntax.Syntax(cfg.to_yaml(), "yaml"))
cfg.output_dir.joinpath("config.yaml").write_text(cfg.to_yaml())


db_path = Path(os.environ["CLEF_IP_LOCATION"]) / "patents_v3.db"
train_topics_path = Path(os.environ["CLEF_IP_LOCATION"]) / "02_topics" / "training-pac" / "clef-ip-2011_PACTraining" / "qrels.txt"

engine = sqlm.create_engine(f"sqlite:///{db_path}")

def fetch_candidate_patent(patent_number: str) -> str:
    with sqlm.Session(engine) as session:
        patent = session.exec(
            sqlm.select(patent_dataset.Patent).where(patent_dataset.Patent.number == patent_number)
        ).first()
        if patent:
            return patent_dataset.extract_query_text(patent, search_columns=["title", "abstract", "claims"], kclaims=1)
        return ""

def fetch_topic_patent(patent_number: str) -> str:
    try:
        topic_file = next(train_topics_path.parent.glob(f"files/{patent_number}.xml"))
        topic_patent = patent_dataset.parse_patent([topic_file])[0]
        return patent_dataset.extract_query_text(topic_patent, search_columns=["title", "abstract", "claims"], kclaims=1)
    except StopIteration:
        logger.warning(f"Topic patent {patent_number} not found.")
        return ""

# --- 2. Data Preparation & Prompting ---
# Qwen Embeddings require specific instruction templates.
# We wrap the patent text in these prompts.
def apply_qwen_template(examples):
    # Retrieve raw texts
    q_text = fetch_topic_patent(examples["anchor"])
    pos_text = fetch_candidate_patent(examples["positive"])    
    neg1_text = fetch_candidate_patent(examples["negatives"][0])  
    neg2_text = fetch_candidate_patent(examples["negatives"][1])

    # 2. Safety Check (Filter empty)
    if not (q_text and pos_text and neg1_text and neg2_text):
        return {"anchor": "", "positive": "", "negative1": "", "negative2": ""}

    # 3. Apply Qwen Instruct Templates
    # Anchor gets the "Query" prompt
    anchor = f"Instruct: Given a patent query, retrieve relevant patent documents.\nQuery: {q_text}\n"

    return {
        "anchor": anchor, 
        "positive": pos_text, 
        "negative1": neg1_text, 
        "negative2": neg2_text
    }

logger.info("Loading Data...")
raw_data = json.load(open("data/mnrl_finetuning_data.json", "r"))
dataset = Dataset.from_list(raw_data)


# MultiNegativeRankingLoss 
dataset = dataset.map(apply_qwen_template)
# Filter out failures (empty strings)
dataset = dataset.filter(
    lambda x: len(x["anchor"]) > 10 and 
              len(x["positive"]) > 10 and 
              len(x["negative1"]) > 10 and 
              len(x["negative2"]) > 10
) 

"""### Contrastive loss
dataset = dataset.map(
    lambda x: {
        "sentence1": fetch_topic_patent(x["sentence1"]),
        "sentence2": fetch_candidate_patent(x["sentence2"])
    }
)"""

# --- 3. Load Model with Unsloth ---
logger.info(f"Loading Unsloth Model: {MODEL_NAME}")
model, tokenizer = FastModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None, 
    load_in_4bit=False,
)

# Enable QLoRA adapters
model = FastModel.get_peft_model(
    model,
    r=cfg.lora_r,
    target_modules=cfg.lora_target_modules,
    lora_alpha=cfg.lora_alpha,
    lora_dropout=cfg.lora_dropout,
    max_seq_length=cfg.max_seq_length,
    bias="none",
    use_gradient_checkpointing="unsloth", # Critical for large context
)

# --- 4. Initialize Sentence Transformer Wrapper ---
# We define the pooling strategy MANUALLY for Qwen (Last Token Pooling is standard for Decoder embeddings)
logger.info("Initializing SentenceTransformer Wrapper...")

embedding_model = Transformer(model_name_or_path=cfg.model_name, 
                              max_seq_length=cfg.max_seq_length)
# Re-assign the Unsloth model to the Transformer wrapper
embedding_model.auto_model = model
embedding_model.tokenizer = tokenizer

# Qwen uses EOS token pooling (Last token), NOT Mean pooling
pooling_model = Pooling(
    embedding_model.get_word_embedding_dimension(),
    pooling_mode=cfg.pooling_mode, # <--- CRITICAL CHANGE for Qwen
    pooling_mode_mean_tokens=False,
    pooling_mode_cls_token=False,
    pooling_mode_max_tokens=False,
)

st_model = SentenceTransformer(modules=[embedding_model, pooling_model])

# --- 5. Loss & Training ---
# Use MultipleNegativesRankingLoss for (Query, Positive) pairs.
# It automatically uses other samples in the batch as negatives (In-Batch Negatives).
loss = MultipleNegativesRankingLoss(st_model)
#loss = ContrastiveLoss(st_model)
args = SentenceTransformerTrainingArguments(
    output_dir=cfg.output_dir,
    num_train_epochs=cfg.epochs,
    per_device_train_batch_size=cfg.batch_size,  # Very small batch size for 4B/7B model + Long Context
    gradient_accumulation_steps=cfg.batch_size,  # Simulate larger batch (Effective BS = 16)
    learning_rate=cfg.learning_rate,
    warmup_steps=cfg.warmup_steps,
    bf16=torch.cuda.is_bf16_supported(),
    #fp16=not torch.cuda.is_bf16_supported(),
    logging_steps=10,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=2,
    optim=cfg.optimizer, # Use 8-bit optimizer to save more VRAM
)

trainer = SentenceTransformerTrainer(
    model=st_model,
    train_dataset=dataset,
    loss=loss,
    args=args,
)

# --- 6. Train ---
logger.info("Starting Training...")
trainer.train()

# --- 7. Save ---
logger.info("Saving Model...")
# Save the adapter
st_model.save_pretrained(cfg.output_dir)

# Note: Unsloth models inside ST wrappers can be tricky to save completely.
# It is often safer to save the adapter explicitly via the internal PEFT model:
model.save_pretrained(f"{cfg.output_dir}/model")
tokenizer.save_pretrained(f"{cfg.output_dir}/model")

logger.info(f"Training Complete. Saved to {cfg.output_dir}")