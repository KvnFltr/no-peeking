"""Zero-shot Yes/No scoring of any Hugging Face causal LM with one forward pass per filing.

P(up) = P(Yes) / (P(Yes) + P(No)) from the next-token distribution after the chat prompt. ``mass`` =
P(Yes) + P(No) tells how much the model actually follows the answer format.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence

import numpy as np

log = logging.getLogger(__name__)

YES_WORDS = ("Yes", " Yes", "yes", " yes", "YES")
NO_WORDS = ("No", " No", "no", " no", "NO")


def load_model(model_id: str, device: str | None = None, dtype: str = "auto", adapter: str | None = None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if device in (None, "auto"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
    kwargs: dict = {}
    if dtype == "4bit":
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16
        )
        kwargs["device_map"] = "auto"
    else:
        if dtype == "auto":
            if device == "cpu":
                torch_dtype = torch.float32
            else:
                torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            torch_dtype = getattr(torch, dtype)
        kwargs["dtype"] = torch_dtype
        if device == "cuda" and torch.cuda.device_count() > 1:
            kwargs["device_map"] = "auto"
    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    if "device_map" not in kwargs:
        model.to(device)
    model.eval()
    return tok, model, device


class YesNoScorer:
    def __init__(
        self,
        model_id: str,
        device: str | None = None,
        dtype: str = "auto",
        max_input_tokens: int = 768,
        adapter: str | None = None,
    ) -> None:
        import torch

        self.torch = torch
        self.tok, self.model, self.device = load_model(model_id, device, dtype, adapter)
        self.max_input_tokens = max_input_tokens
        self.yes_ids = self._first_ids(YES_WORDS)
        self.no_ids = self._first_ids(NO_WORDS)
        if set(self.yes_ids) & set(self.no_ids):
            raise ValueError("Yes/No first tokens overlap for this tokenizer")
        if self.device == "cpu":
            torch.set_num_threads(max(1, torch.get_num_threads()))

    def _first_ids(self, words: Sequence[str]) -> list[int]:
        ids = {
            self.tok.encode(w, add_special_tokens=False)[0]
            for w in words
            if self.tok.encode(w, add_special_tokens=False)
        }
        return sorted(ids)

    def render(self, messages: list[dict[str, str]]) -> list[int]:
        if self.tok.chat_template:
            text = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            text = "\n\n".join(m["content"] for m in messages) + "\nAnswer:"
        return self.tok.encode(text, add_special_tokens=False)

    def encode(self, make_messages: Callable[[int], list[dict[str, str]]]) -> list[int]:
        """Shrink the filing text (not the question) until the prompt fits ``max_input_tokens``."""
        chars = self.max_input_tokens * 4
        ids = self.render(make_messages(chars))
        for _ in range(8):
            if len(ids) <= self.max_input_tokens:
                return ids
            chars = max(200, int(chars * self.max_input_tokens / len(ids) * 0.95))
            ids = self.render(make_messages(chars))
        return ids

    def score(self, batch_ids: list[list[int]], batch_size: int = 4) -> tuple[np.ndarray, np.ndarray]:
        torch = self.torch
        order = np.argsort([len(x) for x in batch_ids])
        p_up = np.empty(len(batch_ids))
        mass = np.empty(len(batch_ids))
        pad = self.tok.pad_token_id
        dev = next(self.model.parameters()).device
        with torch.no_grad():
            for i in range(0, len(order), batch_size):
                idx = order[i : i + batch_size]
                seqs = [batch_ids[j] for j in idx]
                width = max(len(s) for s in seqs)
                input_ids = torch.tensor([[pad] * (width - len(s)) + s for s in seqs], device=dev)
                attn = torch.tensor([[0] * (width - len(s)) + [1] * len(s) for s in seqs], device=dev)
                pos = (attn.cumsum(-1) - 1).clamp(min=0)
                logits = self.model(input_ids=input_ids, attention_mask=attn, position_ids=pos).logits[
                    :, -1, :
                ]
                probs = logits.float().softmax(-1)
                py = probs[:, self.yes_ids].sum(-1)
                pn = probs[:, self.no_ids].sum(-1)
                p_up[idx] = (py / (py + pn).clamp(min=1e-12)).cpu().numpy()
                mass[idx] = (py + pn).cpu().numpy()
        return p_up, mass
