from __future__ import annotations

from pathlib import Path
import random

import torch

from gopt.config import CompileDatasetConfig, SamplingConfig, TrainingConfig
from gopt.data import RecordTokenizer, compile_corpus, load_compiled_corpus, load_corpus, sample_batch
from gopt.go_bpe import render_bpe_records, tokenize_prompt
from gopt.model import GPTLanguageModel


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(
    config: TrainingConfig,
    tokenizer: RecordTokenizer,
    device: torch.device,
) -> GPTLanguageModel:
    if config.d_model % config.n_heads != 0:
        raise ValueError(
            f"d_model={config.d_model} must be divisible by n_heads={config.n_heads}"
        )

    model = GPTLanguageModel(
        vocab_size=tokenizer.vocab_size,
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        block_size=config.block_size,
        dropout=config.dropout,
    )
    return model.to(device)


def _upgrade_state_dict_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Map older checkpoint module names to the current, more descriptive names."""

    if any(key.startswith("transformer_blocks.") for key in state_dict):
        return state_dict

    replacements = [
        ("blocks.", "transformer_blocks."),
        (".ln1.", ".attention_norm."),
        (".attn.", ".attention."),
        (".ln2.", ".feed_forward_norm."),
        (".ffwd.", ".feed_forward."),
        ("ln_f.", "final_norm."),
        (".tril", ".causal_mask"),
        (".key.", ".key_proj."),
        (".query.", ".query_proj."),
        (".value.", ".value_proj."),
    ]

    upgraded: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        new_key = key
        for old, new in replacements:
            new_key = new_key.replace(old, new)
        upgraded[new_key] = value
    return upgraded


def _load_checkpoint(path: str) -> dict[str, object]:
    return torch.load(path, map_location="cpu")


def _build_checkpoint_payload(
    *,
    model: GPTLanguageModel,
    optimizer: torch.optim.Optimizer,
    step: int,
    tokenizer: RecordTokenizer,
    config: TrainingConfig,
    best_val_loss: float | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "step": step,
        "vocab": tokenizer.stoi,
        "config": {
            "d_model": config.d_model,
            "n_heads": config.n_heads,
            "n_layers": config.n_layers,
            "dropout": config.dropout,
            "block_size": config.block_size,
        },
    }
    if best_val_loss is not None:
        payload["best_val_loss"] = best_val_loss
    return payload


def _best_checkpoint_path(checkpoint_path: Path) -> Path:
    return checkpoint_path.with_name(f"{checkpoint_path.stem}.best{checkpoint_path.suffix}")


def _checkpoint_model_config(checkpoint: dict[str, object]) -> dict[str, int | float]:
    return checkpoint["config"]  # type: ignore[return-value]


def _assert_resume_compatible(config: TrainingConfig, checkpoint: dict[str, object]) -> None:
    model_config = _checkpoint_model_config(checkpoint)
    expected = {
        "d_model": config.d_model,
        "n_heads": config.n_heads,
        "n_layers": config.n_layers,
        "dropout": config.dropout,
        "block_size": config.block_size,
    }
    for key, value in expected.items():
        if model_config.get(key) != value:
            raise ValueError(
                f"checkpoint {key}={model_config.get(key)!r} does not match "
                f"requested {key}={value!r}"
            )


@torch.no_grad()
def estimate_loss(
    model: GPTLanguageModel,
    *,
    train_tokens: torch.Tensor,
    val_tokens: torch.Tensor,
    batch_size: int,
    block_size: int,
    eval_batches: int,
    device: torch.device,
    rng: random.Random,
) -> dict[str, float]:
    model.eval()
    losses: dict[str, float] = {}
    for split_name, tokens in (("train", train_tokens), ("val", val_tokens)):
        split_loss = 0.0
        for _ in range(eval_batches):
            x_batch, y_batch = sample_batch(
                tokens,
                batch_size=batch_size,
                block_size=block_size,
                rng=rng,
            )
            _, loss = model(x_batch.to(device), y_batch.to(device))
            assert loss is not None
            split_loss += loss.item()
        losses[split_name] = split_loss / eval_batches
    model.train()
    return losses


def train(config: TrainingConfig) -> None:
    # Build the full training state: text corpus, device, RNG, model, optimizer.
    if config.compiled_dataset_path and Path(config.compiled_dataset_path).exists():
        corpus = load_compiled_corpus(config.compiled_dataset_path)
    else:
        corpus = load_corpus(config.dataset_path, config.val_dataset_path)
    device = resolve_device(config.device)
    rng = random.Random(config.seed)

    torch.manual_seed(config.seed)

    model = build_model(config, corpus.tokenizer, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    start_step = 0
    best_val_loss: float | None = None

    if config.resume:
        checkpoint_path = Path(config.checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"resume checkpoint not found: {checkpoint_path}")

        checkpoint = _load_checkpoint(str(checkpoint_path))
        _assert_resume_compatible(config, checkpoint)
        model.load_state_dict(_upgrade_state_dict_keys(checkpoint["model_state"]))  # type: ignore[index]

        optimizer_state = checkpoint.get("optimizer_state")
        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)  # type: ignore[arg-type]

        start_step = int(checkpoint.get("step", 0))
        stored_best = checkpoint.get("best_val_loss")
        if stored_best is not None:
            best_val_loss = float(stored_best)
        print(f"resuming from step {start_step}")

    checkpoint_path = Path(config.checkpoint_path)
    best_checkpoint_path = _best_checkpoint_path(checkpoint_path)
    end_step = start_step + config.max_steps
    for step in range(start_step, end_step):
        # Periodically estimate loss on fresh train/validation batches.
        if step % config.eval_interval == 0 or step == end_step - 1:
            losses = estimate_loss(
                model,
                train_tokens=corpus.train_tokens,
                val_tokens=corpus.val_tokens,
                batch_size=config.batch_size,
                block_size=config.block_size,
                eval_batches=config.eval_batches,
                device=device,
                rng=rng,
            )
            print(
                f"step {step:04d} "
                f"train_loss={losses['train']:.4f} "
                f"val_loss={losses['val']:.4f}"
            )
            val_loss = losses["val"]
            if best_val_loss is None or val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    _build_checkpoint_payload(
                        model=model,
                        optimizer=optimizer,
                        step=step,
                        tokenizer=corpus.tokenizer,
                        config=config,
                        best_val_loss=best_val_loss,
                    ),
                    best_checkpoint_path,
                )
                print(
                    f"saved best checkpoint to {best_checkpoint_path} "
                    f"(val_loss={best_val_loss:.4f})"
                )

        # Sample a batch of token windows and train on next-token prediction.
        x_batch, y_batch = sample_batch(
            corpus.train_tokens,
            batch_size=config.batch_size,
            block_size=config.block_size,
            rng=rng,
        )
        _, loss = model(x_batch.to(device), y_batch.to(device))
        assert loss is not None

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # Save enough metadata to reconstruct the exact same model for sampling later.
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        _build_checkpoint_payload(
            model=model,
            optimizer=optimizer,
            step=end_step,
            tokenizer=corpus.tokenizer,
            config=config,
            best_val_loss=best_val_loss,
        ),
        checkpoint_path,
    )
    print(f"saved checkpoint to {checkpoint_path}")


def compile_dataset(config: CompileDatasetConfig) -> None:
    compile_corpus(
        train_path=config.train_dataset_path,
        val_path=config.val_dataset_path,
        output_path=config.output_path,
    )
    print(f"compiled dataset to {config.output_path}")


def sample(config: SamplingConfig) -> None:
    print(sample_text(config))


def _brace_depth(records: list[str]) -> int:
    opens = sum(1 for record in records if record == "{")
    closes = sum(1 for record in records if record == "}")
    return opens - closes


def _prompt_has_open_block(prompt: str) -> bool:
    if not prompt:
        return False
    return prompt.count("{") > prompt.count("}")


def _should_stop_generation(prompt_records: list[str], generated_records: list[str]) -> bool:
    prompt_depth = max(0, _brace_depth(prompt_records))
    closes = sum(1 for record in generated_records if record == "}")
    generated_depth = prompt_depth + _brace_depth(generated_records)
    return generated_depth <= 0 and closes > 0


def sample_text(config: SamplingConfig) -> str:
    checkpoint = torch.load(config.checkpoint_path, map_location="cpu")
    stoi = checkpoint["vocab"]
    tokenizer = RecordTokenizer(stoi=stoi, itos={i: record for record, i in stoi.items()})
    model_config = checkpoint["config"]

    device = resolve_device("auto")
    model = GPTLanguageModel(
        vocab_size=tokenizer.vocab_size,
        d_model=model_config["d_model"],
        n_heads=model_config["n_heads"],
        n_layers=model_config["n_layers"],
        block_size=model_config["block_size"],
        dropout=model_config["dropout"],
    ).to(device)
    model.load_state_dict(_upgrade_state_dict_keys(checkpoint["model_state"]))
    model.eval()

    prompt_records = tokenize_prompt(config.prompt, config.bpe_model_path) if config.prompt else []
    prompt_tokens = tokenizer.encode(prompt_records) if prompt_records else [0]
    idx = torch.tensor([prompt_tokens], dtype=torch.long, device=device)

    if _prompt_has_open_block(config.prompt):
        for _ in range(config.max_new_tokens):
            output = model.generate(
                idx,
                max_new_tokens=1,
                temperature=config.temperature,
                top_k=config.top_k,
            )
            idx = output
            generated_records = tokenizer.decode(output[0].tolist())
            suffix_records = generated_records[len(prompt_records) :]
            if _should_stop_generation(prompt_records, suffix_records):
                break
    else:
        output = model.generate(
            idx,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_k=config.top_k,
        )
        generated_records = tokenizer.decode(output[0].tolist())

    return render_bpe_records(generated_records)
