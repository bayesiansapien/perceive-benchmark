"""
GPU model adapter for Phase 3 evaluation via HuggingFace transformers.
Handles: Qwen3.5-VL-4B, Phi-4-Vision, InternVL3-8B, Qwen3.5-35B-A3B (MoE)
Designed to run on DGX server with A100 80GB.

No vLLM dependency: uses transformers directly for ABI compatibility with
NVIDIA custom PyTorch 2.8.0.
"""
from __future__ import annotations

import base64
import io
import os
import re
import time
import logging
from typing import Optional

import torch

from .base_adapter import BaseModelAdapter

log = logging.getLogger(__name__)

EVAL_SYSTEM_PROMPT = (
    "Answer the question about the given document image. "
    "Give a short, precise answer. "
    "If the answer is a number, return just the number. "
    "If it is a name or short text, return just that text. "
    "Be concise."
)

# HF model IDs per yaml_key
MODEL_HF_IDS = {
    "a1_qwen35vl4b":  "Qwen/Qwen3.5-VL-4B-Instruct",
    "a3_phi4vision":   "microsoft/Phi-4-vision-instruct",
    "b2_internvl3":    "OpenGVLab/InternVL3-8B",
    "b4_qwen35b_moe":  "Qwen/Qwen3.5-122B-A10B-FP8",
}

# Max sequence lengths per model (to fit in VRAM)
MODEL_MAX_LEN = {
    "a1_qwen35vl4b":  4096,
    "a3_phi4vision":   4096,
    "b2_internvl3":    4096,
    "b4_qwen35b_moe":  8192,
}

# Models that support thinking (native)
THINKING_MODELS = {"b4_qwen35b_moe"}

# ImageNet normalization constants for InternVL3
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)


class GPUAdapter(BaseModelAdapter):
    """
    GPU adapter using HuggingFace transformers for inference.
    The model and processor are loaded once via load_model() and stored on
    the instance. Call load_model() once per model, then use adapter.call()
    for each sample.
    """

    def __init__(
        self,
        yaml_key: str,
        model_cfg: dict,
        budget_level: str,
        model=None,
        processor=None,
    ):
        super().__init__(yaml_key, model_cfg, budget_level)
        self.model = model
        self.processor = processor  # AutoProcessor, AutoTokenizer, or tokenizer
        self.hf_id = MODEL_HF_IDS.get(yaml_key, model_cfg.get("hf_id", ""))
        self.supports_thinking = yaml_key in THINKING_MODELS

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def call(self, image_b64: str, query: str) -> dict:
        """
        Run inference on one sample.

        Args:
            image_b64: Base64-encoded image bytes.
            query: The question string.

        Returns:
            {
                "answer": str,
                "input_tokens": int,
                "output_tokens": int,
                "reasoning_tokens": int,
                "latency_ms": int,
            }
        """
        if self.model is None:
            raise RuntimeError(
                "GPUAdapter.model is None. Use GPUAdapter.load_model() first."
            )

        img_bytes = base64.b64decode(image_b64)
        pil_image = _bytes_to_pil(img_bytes)

        dispatch = {
            "a1_qwen35vl4b":  self._call_qwen35vl,
            "a3_phi4vision":   self._call_phi4vision,
            "b2_internvl3":    self._call_internvl3,
            "b4_qwen35b_moe":  self._call_qwen35b_moe,
        }

        fn = dispatch.get(self.yaml_key)
        if fn is None:
            raise ValueError(
                f"Unknown yaml_key '{self.yaml_key}'. "
                f"Supported: {list(dispatch.keys())}"
            )

        return fn(pil_image, query)

    @classmethod
    def load_model(
        cls,
        yaml_key: str,
        model_cfg: dict,
        budget_level: str,
        gpu_memory_utilization: float = 0.90,
        tensor_parallel_size: int = 1,
    ) -> "GPUAdapter":
        """
        Load model + processor using HuggingFace transformers and return
        a fully initialised GPUAdapter.

        gpu_memory_utilization and tensor_parallel_size are accepted for
        interface compatibility; gpu_memory_utilization is used to set
        PYTORCH_CUDA_ALLOC_CONF when < 1.0.
        """
        hf_id = MODEL_HF_IDS.get(yaml_key, model_cfg.get("hf_id", ""))

        # Propagate memory fraction hint via env var for the CUDA allocator
        if gpu_memory_utilization < 1.0:
            frac = f"{gpu_memory_utilization:.2f}"
            existing = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
            if "max_split_size_mb" not in existing:
                os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
                    f"garbage_collection_threshold:{frac}"
                    + (f",{existing}" if existing else "")
                )

        log.info("Loading %s (%s) via transformers ...", yaml_key, hf_id)
        t0 = time.monotonic()

        loaders = {
            "a1_qwen35vl4b":  _load_qwen35vl,
            "a3_phi4vision":   _load_phi4vision,
            "b2_internvl3":    _load_internvl3,
            "b4_qwen35b_moe":  _load_qwen35b_moe,
        }

        loader = loaders.get(yaml_key)
        if loader is None:
            raise ValueError(
                f"No loader for yaml_key '{yaml_key}'. "
                f"Supported: {list(loaders.keys())}"
            )

        model, processor = loader(hf_id)

        elapsed = time.monotonic() - t0
        log.info("Loaded %s in %.1fs", yaml_key, elapsed)

        return cls(yaml_key, model_cfg, budget_level, model=model, processor=processor)

    # ------------------------------------------------------------------
    # Per-model inference implementations
    # ------------------------------------------------------------------

    def _call_qwen35vl(self, pil_image, query: str) -> dict:
        """Qwen3.5-VL-4B inference."""
        try:
            from qwen_vl_utils import process_vision_info
            _have_qwen_vl_utils = True
        except ImportError:
            log.warning(
                "qwen_vl_utils not installed; falling back to direct PIL image. "
                "Install with: pip install qwen-vl-utils"
            )
            _have_qwen_vl_utils = False

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": query},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        if _have_qwen_vl_utils:
            from qwen_vl_utils import process_vision_info
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text], images=image_inputs, return_tensors="pt"
            ).to(self.model.device)
        else:
            inputs = self.processor(
                text=[text], images=[pil_image], return_tensors="pt"
            ).to(self.model.device)

        n_input = inputs.input_ids.shape[1]

        t0 = time.monotonic()
        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=None,
                do_sample=False,
            )
        latency_ms = int((time.monotonic() - t0) * 1000)

        generated = output_ids[0][n_input:]
        answer = self.processor.decode(generated, skip_special_tokens=True).strip()

        return {
            "answer": answer,
            "input_tokens": n_input,
            "output_tokens": len(generated),
            "reasoning_tokens": 0,
            "latency_ms": latency_ms,
        }

    def _call_phi4vision(self, pil_image, query: str) -> dict:
        """Phi-4-Vision inference."""
        messages = [
            {"role": "system", "content": EVAL_SYSTEM_PROMPT},
            {"role": "user", "content": "<|image_1|>\n" + query},
        ]

        prompt = self.processor.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            prompt, images=[pil_image], return_tensors="pt"
        ).to(self.model.device)

        n_input = inputs["input_ids"].shape[1]

        t0 = time.monotonic()
        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
            )
        latency_ms = int((time.monotonic() - t0) * 1000)

        generated = output_ids[0][n_input:]
        answer = self.processor.tokenizer.decode(
            generated, skip_special_tokens=True
        ).strip()

        return {
            "answer": answer,
            "input_tokens": n_input,
            "output_tokens": len(generated),
            "reasoning_tokens": 0,
            "latency_ms": latency_ms,
        }

    def _call_internvl3(self, pil_image, query: str) -> dict:
        """InternVL3-8B inference."""
        import torchvision.transforms as T
        from torchvision.transforms.functional import InterpolationMode

        transform = T.Compose([
            T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ])

        pixel_values = (
            transform(pil_image)
            .unsqueeze(0)
            .to(torch.bfloat16)
            .to(self.model.device)
        )

        question = f"<image>\n{query}"
        generation_config = {"max_new_tokens": 512, "do_sample": False}

        t0 = time.monotonic()
        with torch.inference_mode():
            response = self.model.chat(
                self.processor,  # tokenizer stored in processor slot
                pixel_values,
                question,
                generation_config=generation_config,
            )
        latency_ms = int((time.monotonic() - t0) * 1000)

        # InternVL.chat() returns a string; token counts approximated via tokenizer
        answer = response.strip() if isinstance(response, str) else str(response).strip()

        encoded_in = self.processor.encode(question, return_tensors="pt")
        encoded_out = self.processor.encode(answer, return_tensors="pt")
        n_input = encoded_in.shape[1]
        n_output = encoded_out.shape[1]

        return {
            "answer": answer,
            "input_tokens": n_input,
            "output_tokens": n_output,
            "reasoning_tokens": 0,
            "latency_ms": latency_ms,
        }

    def _call_qwen35b_moe(self, pil_image, query: str) -> dict:
        """Qwen3.5-122B-A10B-FP8, text-only MoE inference (no image encoder)."""
        enable_thinking = (
            self.supports_thinking and self.budget_level in ("B1", "B3")
        )
        max_new_tokens = self.budget_tokens + 512 if enable_thinking else 512

        messages = [
            {"role": "system", "content": EVAL_SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ]

        template_kwargs = dict(tokenize=False, add_generation_prompt=True)
        if enable_thinking:
            template_kwargs["enable_thinking"] = True

        text = self.processor.apply_chat_template(messages, **template_kwargs)
        inputs = self.processor(text, return_tensors="pt").to(self.model.device)
        n_input = inputs.input_ids.shape[1]

        t0 = time.monotonic()
        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            )
        latency_ms = int((time.monotonic() - t0) * 1000)

        generated = output_ids[0][n_input:]
        raw = self.processor.decode(generated, skip_special_tokens=True)
        answer, reasoning_tokens = _extract_thinking_answer(raw, enable_thinking)

        return {
            "answer": answer,
            "input_tokens": n_input,
            "output_tokens": len(generated),
            "reasoning_tokens": reasoning_tokens,
            "latency_ms": latency_ms,
        }


# ------------------------------------------------------------------
# Model loaders (called once at startup)
# ------------------------------------------------------------------

def _load_qwen35vl(hf_id: str):
    """Load Qwen3.5-VL-4B-Instruct."""
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        hf_id,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(hf_id, trust_remote_code=True)
    return model, processor


def _load_phi4vision(hf_id: str):
    """Load Phi-4-vision-instruct."""
    from transformers import AutoModelForCausalLM, AutoProcessor

    model = AutoModelForCausalLM.from_pretrained(
        hf_id,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
        _attn_implementation="eager",
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(hf_id, trust_remote_code=True)
    return model, processor


def _load_internvl3(hf_id: str):
    """Load InternVL3-8B. Returns (model, tokenizer), tokenizer stored in processor slot."""
    from transformers import AutoModel, AutoTokenizer

    model = AutoModel.from_pretrained(
        hf_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
    # Store tokenizer in the "processor" slot for uniform access
    return model, tokenizer


def _load_qwen35b_moe(hf_id: str):
    """Load Qwen3.5-122B-A10B-FP8, text-only MoE, FP8 quantized."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        hf_id,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    # Return tokenizer as processor (text-only model)
    processor = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
    return model, processor


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _bytes_to_pil(img_bytes: bytes):
    """Convert raw bytes to PIL Image."""
    from PIL import Image
    return Image.open(io.BytesIO(img_bytes)).convert("RGB")


def _extract_thinking_answer(raw_output: str, thinking_enabled: bool) -> tuple[str, int]:
    """
    Split raw model output into (answer, reasoning_tokens).
    For Qwen thinking models: strips <think>...</think> blocks.
    Returns (answer_text, estimated_reasoning_tokens).
    """
    if not thinking_enabled:
        return raw_output.strip(), 0

    think_pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL)
    thinking_content = " ".join(think_pattern.findall(raw_output))
    answer = think_pattern.sub("", raw_output).strip()

    # Estimate reasoning tokens (~4 chars per token)
    reasoning_tokens = len(thinking_content) // 4

    return answer, reasoning_tokens
