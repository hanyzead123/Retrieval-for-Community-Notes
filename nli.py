"""
NLI-based support scoring using a DeBERTa-v3-base model fine-tuned on MNLI.

Given a query q (post + community note) and a candidate passage p (excerpt),
computes how well p supports the claim in q via entailment probability.
"""

from __future__ import annotations

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Public MNLI model (DeBERTa v3 base); microsoft/deberta-v3-base is pretrained-only, no NLI labels
MODEL_ID = "MoritzLaurer/DeBERTa-v3-base-mnli"
ENTAILMENT_LABEL = "ENTAILMENT"
CONTRADICTION_LABEL = "CONTRADICTION"
DEFAULT_MAX_LENGTH = 512
DEFAULT_TEMPERATURE = 1.0


def _infer_device() -> str:
    """Pick best available device: CUDA > MPS (Apple Silicon) > CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _get_entailment_id(model) -> int:
    """Return the label id for ENTAILMENT from the model config."""
    id2label = getattr(model.config, "id2label", {})
    if not id2label and hasattr(model.config, "id2label"):
        id2label = dict(model.config.id2label)
    for idx, name in id2label.items():
        if (name or "").upper() == ENTAILMENT_LABEL:
            return int(idx)
    if hasattr(model.config, "label2id"):
        for key in (ENTAILMENT_LABEL, "entailment"):
            if key in model.config.label2id:
                return int(model.config.label2id[key])
    return 2  # fallback: MNLI models typically use 2 for ENTAILMENT


class NLISupportScorer:
    """
    Scores how well a candidate passage supports a claim (query) using
    an MNLI-finetuned model (e.g. DeBERTa-v3-base-mnli). Support score = P(entailment | premise=passage, hypothesis=query).
    """

    def __init__(
        self,
        model_id: str = MODEL_ID,
        device: str | None = None,
        max_length: int = DEFAULT_MAX_LENGTH,
        temperature: float = DEFAULT_TEMPERATURE,
    ):
        self.model_id = model_id
        self.max_length = max_length
        self.device = device or _infer_device()
        self._model = None
        self._tokenizer = None
        self._entailment_id = None
        self._contradiction_id = None
        self.temperature = float(temperature)

    @property
    def model(self):
        if self._model is None:
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.model_id
            ).to(self.device)
            self._model.eval()
            # Read label mapping from config and cache entailment/contradiction ids
            id2label = getattr(self._model.config, "id2label", None)
            if id2label is None and hasattr(self._model.config, "label2id"):
                # create id2label if only label2id is present
                id2label = {v: k for k, v in dict(self._model.config.label2id).items()}

            # id2label expected mapping of id->label
            self._entailment_id = None
            self._contradiction_id = None
            if id2label:
                for idx, name in id2label.items():
                    if (name or "").upper() == ENTAILMENT_LABEL:
                        self._entailment_id = int(idx)
                    if (name or "").upper() == CONTRADICTION_LABEL:
                        self._contradiction_id = int(idx)

            # Fallbacks if mapping isn't available
            if self._entailment_id is None:
                self._entailment_id = _get_entailment_id(self._model)
            if self._contradiction_id is None:
                # typical MNLI ordering: contradiction=0, neutral=1, entailment=2
                self._contradiction_id = 0 if self._entailment_id != 0 else 1

            # Debug log of mapping
            try:
                print(f"NLI label mapping (id2label): {self._model.config.id2label}")
            except Exception:
                pass
        return self._model

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        return self._tokenizer

    def score(self, query: str, passage: str) -> float:
        """
        Compute support score: how well `passage` supports the claim in `query`.

        In NLI terms: premise = passage (evidence), hypothesis = query (claim).
        Returns P(entailment), in [0, 1].

        Args:
            query: Concatenation of original post + community note (the claim).
            passage: Candidate excerpt/source passage.

        Returns:
            Support score in [0, 1].
        """
        logits = self.score_logits(query, passage)
        probs = self._apply_temperature_and_softmax(logits, self.temperature)
        return float(probs[self._entailment_id])

    def score_logits(self, query: str, passage: str) -> list[float]:
        """
        Return raw logits for a single (query, passage) pair.
        """
        inputs = self.tokenizer(
            passage,
            query,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=True,
        ).to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        return logits[0].cpu().numpy().tolist()

    def score_batch_logits(self, pairs: list[tuple[str, str]]) -> list[list[float]]:
        """Return raw logits for multiple (query, passage) pairs."""
        if not pairs:
            return []
        premises = [p for _, p in pairs]
        hypotheses = [q for q, _ in pairs]
        inputs = self.tokenizer(
            premises,
            hypotheses,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=True,
        ).to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        return logits.cpu().numpy().tolist()

    def _apply_temperature_and_softmax(self, logits: list[float] | list[list[float]], temperature: float = 1.0):
        import numpy as _np

        arr = _np.array(logits)
        if arr.ndim == 1:
            vals = arr / float(temperature)
            ex = _np.exp(vals - vals.max())
            probs = ex / ex.sum()
            return probs.tolist()
        else:
            vals = arr / float(temperature)
            ex = _np.exp(vals - vals.max(axis=1, keepdims=True))
            probs = ex / ex.sum(axis=1, keepdims=True)
            return probs.tolist()

    def decide_label(self, logits: list[float], temperature: float | None = None, thresholds: dict | None = None) -> dict:
        """
        Decide SUPPORT/NEUTRAL/CONTRADICT using calibrated probabilities, margin, and thresholds.

        Returns dict with: label, entail_prob, contra_prob, margin, logits
        """
        if temperature is None:
            temperature = self.temperature
        probs = self._apply_temperature_and_softmax(logits, temperature)
        entail_p = float(probs[self._entailment_id])
        contra_p = float(probs[self._contradiction_id])
        import numpy as _np

        margin = float(_np.array(logits)[self._entailment_id] - _np.array(logits)[self._contradiction_id])

        # default thresholds (can be tuned)
        thr = thresholds or {
            "support_prob": 0.45,
            "support_prob_lo": 0.30,
            "support_margin": 1.0,
            "neutral_lo": 0.15,
            "contra_margin": 0.8,
        }

        label = "CONTRADICT"
        if entail_p >= thr["support_prob"] or (entail_p >= thr["support_prob_lo"] and margin >= thr["support_margin"]):
            label = "SUPPORT"
        elif entail_p >= thr["neutral_lo"]:
            label = "NEUTRAL"
        else:
            label = "CONTRADICT"

        return {
            "label": label,
            "entail_prob": entail_p,
            "contra_prob": contra_p,
            "margin": margin,
            "logits": logits,
        }

    def set_temperature(self, temp: float):
        self.temperature = float(temp)

    def fit_temperature(self, logits_list: list[list[float]], targets: list[int]) -> float:
        """
        Placeholder for fitting temperature on validation set (requires labels).
        Implement optimization (e.g., minimize NLL) externally and call `set_temperature`.
        """
        raise NotImplementedError("Temperature fitting requires a labeled calibration set")

    def score_batch(self, pairs: list[tuple[str, str]]) -> list[float]:
        """
        Compute support scores for multiple (query, passage) pairs.

        Args:
            pairs: List of (query, passage) tuples.

        Returns:
            List of support scores in [0, 1].
        """
        if not pairs:
            return []
        premises = [p for _, p in pairs]
        hypotheses = [q for q, _ in pairs]
        inputs = self.tokenizer(
            premises,
            hypotheses,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=True,
        ).to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)
        return [probs[i, self._entailment_id].item() for i in range(len(pairs))]


# Convenience: lazy-loaded singleton for simple use from other modules
_scorer: NLISupportScorer | None = None


def get_scorer(
    model_id: str = MODEL_ID,
    device: str | None = None,
    max_length: int = DEFAULT_MAX_LENGTH,
    temperature: float = DEFAULT_TEMPERATURE,
) -> NLISupportScorer:
    """Return a shared NLISupportScorer instance (lazy-loaded)."""
    global _scorer
    if _scorer is None:
        _scorer = NLISupportScorer(model_id=model_id, device=device, max_length=max_length, temperature=temperature)
    return _scorer


def support_score(query: str, passage: str, **scorer_kwargs) -> float:
    """
    One-off support score for (query, passage). Uses shared scorer if available.

    Args:
        query: Post + community note (claim).
        passage: Candidate excerpt.
        **scorer_kwargs: Passed to NLISupportScorer if creating one.

    Returns:
        Support score in [0, 1].
    """
    return get_scorer(**scorer_kwargs).score(query, passage)
