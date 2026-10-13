"""Sequential model loading and completion-only LoRA training."""

import gc
import json
import time

import joblib
import psutil
import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed

from .core import CFG, ROOT, messages, parse, read, save


def classic_train():
    rows = read("data/train.json")
    model = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=30000),
        LogisticRegression(max_iter=1000, random_state=CFG["seed"]),
    )
    start = time.perf_counter()
    model.fit([r["text"] for r in rows], [r["category"] for r in rows])
    (ROOT / "artifacts").mkdir(exist_ok=True)
    joblib.dump(model, ROOT / "artifacts/classic.joblib")
    save(ROOT / "reports/classic_training.json", {"seconds": time.perf_counter() - start, "n": len(rows)})


class Engine:
    def __init__(self, kind, adapter="adapter_smoke"):
        self.kind = kind
        self.labels = read("data/manifest.json")["labels"]
        self.variant = (
            read("artifacts/prompt.json")["variant"] if (ROOT / "artifacts/prompt.json").exists() else 0
        )
        if kind == "classic":
            self.model = joblib.load(ROOT / "artifacts/classic.joblib")
            return
        if kind == "lora" and not (ROOT / f"artifacts/{adapter}/adapter_config.json").exists():
            raise FileNotFoundError("Adaptateur absent : lancez l'entraînement LoRA.")
        torch.set_num_threads(CFG["threads"])
        pin = (
            read("artifacts/model_revision.json")
            if (ROOT / "artifacts/model_revision.json").exists()
            else read("configs/model_revision.json")
        )
        self.tokenizer = AutoTokenizer.from_pretrained(CFG["model"], revision=pin.get("revision"))
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            CFG["model"], revision=pin.get("revision"), dtype=torch.float32
        )
        save(
            ROOT / "artifacts/model_revision.json",
            {"model": CFG["model"], "revision": self.model.config._commit_hash},
        )
        if kind == "lora":
            self.model = PeftModel.from_pretrained(self.model, ROOT / f"artifacts/{adapter}")
        self.model.eval()

    def prompt_tokens(self, text, variant=None):
        ids = self.tokenizer.encode(text, add_special_tokens=False)[: CFG["max_text_tokens"]]
        bounded = self.tokenizer.decode(ids)
        return self.tokenizer.apply_chat_template(
            messages(bounded, self.labels, self.variant if variant is None else variant),
            tokenize=True,
            add_generation_prompt=True,
        )

    def predict(self, text):
        start = time.perf_counter()
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Réclamation vide")
        if self.kind == "classic":
            obj = {"category": str(self.model.predict([text])[0]), "summary": None, "justification": None}
            raw, error = json.dumps(obj), None
        else:
            inputs = torch.tensor([self.prompt_tokens(text)])
            with torch.inference_mode():
                generated = self.model.generate(
                    input_ids=inputs,
                    attention_mask=torch.ones_like(inputs),
                    max_new_tokens=CFG["max_new_tokens"],
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            raw = self.tokenizer.decode(generated[0, inputs.shape[1] :], skip_special_tokens=True).strip()
            obj, error = parse(raw, self.labels)
        return {"output": obj, "raw": raw, "error": error, "seconds": time.perf_counter() - start}

    def close(self):
        del self.model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def collate(features):
    length = max(len(f["input_ids"]) for f in features)
    return {
        key: torch.tensor([f[key] + [pad] * (length - len(f[key])) for f in features])
        for key, pad in [("input_ids", 2), ("attention_mask", 0), ("labels", -100)]
    }


def encode(engine, row):
    prefix = engine.prompt_tokens(row["text"])
    target = json.dumps({"category": row["category"], "summary": None, "justification": None})
    suffix = engine.tokenizer.encode(target + engine.tokenizer.eos_token, add_special_tokens=False)
    return {
        "input_ids": prefix + suffix,
        "attention_mask": [1] * (len(prefix) + len(suffix)),
        "labels": [-100] * len(prefix) + suffix,
    }


def lora_train(smoke=True):
    set_seed(CFG["seed"])
    engine = Engine("base")
    engine.model = get_peft_model(
        engine.model,
        LoraConfig(
            r=CFG["lora_r"],
            lora_alpha=CFG["lora_alpha"],
            lora_dropout=CFG["lora_dropout"],
            target_modules=["q_proj", "v_proj"],
            task_type="CAUSAL_LM",
        ),
    )
    train = read("data/train.json")
    validation = read("data/validation.json")
    if smoke:
        train, validation = train[:8], validation[:3]
    ds = Dataset.from_list([encode(engine, r) for r in train])
    vd = Dataset.from_list([encode(engine, r) for r in validation])
    name = "adapter_smoke" if smoke else "adapter_full"
    output = ROOT / f"artifacts/{name}"
    save(
        output / "provenance.json",
        {
            "dataset": read("data/manifest.json"),
            "prompt": read("artifacts/prompt.json"),
            "base": read("artifacts/model_revision.json"),
            "train_ids": [r["id"] for r in train],
            "validation_ids": [r["id"] for r in validation],
        },
    )
    args = TrainingArguments(
        output_dir=str(output),
        use_cpu=not torch.cuda.is_available(),
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=2,
        max_steps=CFG["smoke_steps"] if smoke else -1,
        num_train_epochs=CFG["full_epochs"],
        learning_rate=CFG["learning_rate"],
        logging_steps=1,
        eval_strategy="steps" if smoke else "epoch",
        eval_steps=CFG["smoke_steps"],
        save_strategy="steps" if smoke else "epoch",
        save_steps=CFG["smoke_steps"],
        save_total_limit=2,
        report_to="none",
        seed=CFG["seed"],
        dataloader_pin_memory=False,
        remove_unused_columns=False,
    )
    trainer = Trainer(model=engine.model, args=args, train_dataset=ds, eval_dataset=vd, data_collator=collate)
    trainable, total = engine.model.get_nb_trainable_parameters()
    start = time.perf_counter()
    trainer.train()
    engine.model.save_pretrained(output)
    engine.tokenizer.save_pretrained(output)
    save(
        output / "run.json",
        {
            "smoke": smoke,
            "n_train": len(train),
            "trainable": trainable,
            "total": total,
            "seconds": time.perf_counter() - start,
            "rss_bytes_end": psutil.Process().memory_info().rss,
            "history": trainer.state.log_history,
            "config": CFG,
            "objective": "completion-only category JSON",
        },
    )
    import matplotlib.pyplot as plt

    losses = [r for r in trainer.state.log_history if "loss" in r]
    plt.plot([r["step"] for r in losses], [r["loss"] for r in losses])
    plt.xlabel("Étape")
    plt.ylabel("Loss entraînement")
    plt.savefig(output / "loss.png")
    plt.close()
    engine.close()
    del trainer
    gc.collect()
    reloaded = Engine("lora", adapter=name)
    result = reloaded.predict(validation[0]["text"])
    save(output / "reload_check.json", result)
    reloaded.close()
    return name
