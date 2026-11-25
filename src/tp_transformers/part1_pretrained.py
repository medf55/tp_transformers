"""Première Partie : modèles préentraînés GPT-2 (génération) et BERT (masquage)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from tp_transformers.utils import configure_hf_environment, get_device, set_seed

configure_hf_environment()

from tokenizers import Tokenizer  # noqa: E402
from tokenizers.models import WordLevel  # noqa: E402
from tokenizers.pre_tokenizers import Whitespace  # noqa: E402
from transformers import (  # noqa: E402
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoTokenizer,
    BertConfig,
    BertForMaskedLM,
    GPT2Config,
    GPT2LMHeadModel,
    PreTrainedTokenizerFast,
)


@dataclass(frozen=True)
class GPT2Result:
    """Résultat d'une génération autorégressive avec GPT-2."""

    prompt: str
    generated_text: str
    input_shape: tuple[int, ...]
    num_new_tokens: int


@dataclass(frozen=True)
class BERTCandidate:
    """Proposition individuelle de BERT pour un token masqué."""

    token: str
    probability: float
    completed_sentence: str


@dataclass(frozen=True)
class BERTPredictionResult:
    """Résultat complet de prédiction de mot masqué avec BERT."""

    original_text: str
    input_shape: tuple[int, ...]
    mask_position: int
    candidates: list[BERTCandidate]


class GPT2Generator:
    """Encapsule GPT-2 préentraîné pour la complétion autorégressive de texte."""

    def __init__(
        self,
        model_name: str = "gpt2",
        *,
        local_files_only: bool = False,
        model: Any | None = None,
        tokenizer: Any | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = get_device()
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(
            model_name, local_files_only=local_files_only
        )
        self.model = model or AutoModelForCausalLM.from_pretrained(
            model_name, local_files_only=local_files_only
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.to(self.device)
        self.model.eval()

    def generate(
        self,
        prompt: str = "Artificial intelligence can help people",
        max_new_tokens: int = 50,
        temperature: float = 0.8,
        top_p: float = 0.9,
        do_sample: bool = True,
        seed: int = 42,
    ) -> GPT2Result:
        """Génère la suite d'une amorce avec contrôle de température et noyau top-p."""
        if not prompt.strip():
            raise ValueError("L'amorce ne doit pas être vide.")
        set_seed(seed)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_shape = tuple(inputs["input_ids"].shape)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        generated_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        num_new_tokens = int(output_ids.shape[1] - inputs["input_ids"].shape[1])
        return GPT2Result(
            prompt=prompt,
            generated_text=generated_text,
            input_shape=input_shape,
            num_new_tokens=num_new_tokens,
        )


class BERTMaskPredictor:
    """Encapsule BERT multilingue pour la prédiction bidirectionnelle de [MASK]."""

    def __init__(
        self,
        model_name: str = "bert-base-multilingual-cased",
        *,
        local_files_only: bool = False,
        model: Any | None = None,
        tokenizer: Any | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = get_device()
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(
            model_name, local_files_only=local_files_only
        )
        self.model = model or AutoModelForMaskedLM.from_pretrained(
            model_name, local_files_only=local_files_only
        )
        self.model.to(self.device)
        self.model.eval()

    def predict_masked(self, text: str, top_k: int = 5) -> BERTPredictionResult:
        """Prédit les top_k tokens les plus probables pour un unique [MASK]."""
        mask_tok = self.tokenizer.mask_token
        prepared = text
        if mask_tok not in prepared and "[MASK]" in prepared:
            prepared = prepared.replace("[MASK]", mask_tok)

        inputs = self.tokenizer(prepared, return_tensors="pt").to(self.device)
        mask_positions = (
            inputs["input_ids"][0] == self.tokenizer.mask_token_id
        ).nonzero(as_tuple=True)[0]

        if len(mask_positions) != 1:
            raise ValueError(
                f"Le texte doit contenir exactement un token {mask_tok} "
                f"(trouvé : {len(mask_positions)})."
            )

        mask_pos = int(mask_positions[0].item())
        with torch.no_grad():
            outputs = self.model(**inputs)

        scores = outputs.logits[0, mask_pos, :]
        probs = torch.softmax(scores, dim=-1)
        top_probs, top_indices = torch.topk(probs, k=min(top_k, probs.shape[-1]))

        candidates: list[BERTCandidate] = []
        for prob, idx in zip(top_probs, top_indices):
            token_str = self.tokenizer.decode([int(idx.item())]).strip()
            completed = prepared.replace(mask_tok, token_str, 1)
            candidates.append(
                BERTCandidate(
                    token=token_str,
                    probability=float(prob.item()),
                    completed_sentence=completed,
                )
            )

        return BERTPredictionResult(
            original_text=prepared,
            input_shape=tuple(inputs["input_ids"].shape),
            mask_position=mask_pos,
            candidates=candidates,
        )


def build_offline_demo_gpt2(seed: int = 42) -> GPT2Generator:
    """Construit un mini GPT2LMHeadModel Transformers en mémoire pour tests hors-ligne."""
    set_seed(seed)
    vocab = {
        "<|endoftext|>": 0,
        "[UNK]": 1,
        "Artificial": 2,
        "intelligence": 3,
        "can": 4,
        "help": 5,
        "people": 6,
        "learn": 7,
        "and": 8,
        "build": 9,
        "systems": 10,
        ".": 11,
    }
    tok_obj = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    tok_obj.pre_tokenizer = Whitespace()
    tok_obj.add_special_tokens(["<|endoftext|>", "[UNK]"])
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tok_obj,
        eos_token="<|endoftext|>",
        unk_token="[UNK]",
    )
    config = GPT2Config(
        vocab_size=len(vocab),
        n_positions=64,
        n_embd=32,
        n_layer=1,
        n_head=2,
        bos_token_id=0,
        eos_token_id=0,
    )
    model = GPT2LMHeadModel(config)
    return GPT2Generator("gpt2-offline-demo", model=model, tokenizer=tokenizer)


def build_offline_demo_bert(seed: int = 42) -> BERTMaskPredictor:
    """Construit un mini BertForMaskedLM Transformers en mémoire pour tests hors-ligne."""
    set_seed(seed)
    vocab = {
        "[PAD]": 0,
        "[UNK]": 1,
        "[CLS]": 2,
        "[SEP]": 3,
        "[MASK]": 4,
        "Rabat": 5,
        "est": 6,
        "la": 7,
        "capitale": 8,
        "du": 9,
        "Maroc": 10,
        ".": 11,
        "Paris": 12,
        "de": 13,
        "France": 14,
    }
    tok_obj = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    tok_obj.pre_tokenizer = Whitespace()
    tok_obj.add_special_tokens(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"])
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tok_obj,
        pad_token="[PAD]",
        unk_token="[UNK]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
    )
    config = BertConfig(
        vocab_size=len(vocab),
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=64,
        pad_token_id=0,
    )
    model = BertForMaskedLM(config)
    return BERTMaskPredictor("bert-offline-demo", model=model, tokenizer=tokenizer)
