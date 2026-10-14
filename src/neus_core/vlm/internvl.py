"""InternVL loading, image preprocessing and yes/no proposition detection.

Shared by every detector (NeuS-QA sequential, RT-NeuS KV-cache, ablation
batched). Subclasses only implement ``score``: how the yes/no questions for
one window of frames are run through the model.
"""

import gc
import logging
import math

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

from neus_core.vlm.obj import DetectedObject
from neus_core.vlm.sigmoid import calibrate_sigmoid
from neus_core.vlm.yes_no import YesNoScore

PARSING_RULE_SUBTITLE = (
    "You must only return a Yes or No, and not both, to any question asked. You must not include any other "
    "symbols, information, text, justification in your answer or repeat Yes or No multiple times. For example, "
    "if the question is \"Does the video have the subtitle 'this is very interesting' present in the sequence of "
    "images?\", the answer must only be 'Yes' or 'No'."
)
PARSING_RULE_OBJECT = (
    "You must only return a Yes or No, and not both, to any question asked. You must not include any other "
    "symbols, information, text, justification in your answer or repeat Yes or No multiple times. For example, "
    "if the question is \"Is there a cat present in the sequence of images?\", the answer must only be 'Yes' or 'No'."
)


def detection_prompt(proposition: str) -> str:
    """Yes/no question asked to the VLM for one proposition."""
    if "subtitle" in proposition:
        text = proposition.replace("subtitle_", "").replace("_", " ")
        return (
            f"Does the video have the subtitle '{text}' present in the sequence of images? "
            f"\n[PARSING RULE]: {PARSING_RULE_SUBTITLE}"
        )
    # PULS prefixes propositions that start with a digit with "n_"; it is not part of the text
    if proposition.startswith("n_"):
        proposition = proposition[2:]
    text = proposition.replace("_", " ")
    return f"Is there a '{text}' present in the sequence of images? \n[PARSING RULE]: {PARSING_RULE_OBJECT}"


class InternVLDetector:
    """Loads an InternVL checkpoint and turns yes/no scores into ``DetectedObject``s."""

    def __init__(
        self,
        model_name: str = "InternVL2-8B",
        multi_gpus: bool = False,
        device: int = 0,
        max_patch: int = 1,
    ) -> None:
        self.model_name = model_name
        self._path = f"OpenGVLab/{model_name}"
        self.device = device
        if multi_gpus:
            device_map = split_model(model_name)
        else:
            device_map = assign_device_map(model_name=model_name, manual_gpu_id=device)
        self.model = AutoModel.from_pretrained(
            self._path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            attn_implementation="flash_attention_2",
            trust_remote_code=True,
            device_map=device_map,
        ).eval()
        self.model.apply(self.move_tensors_to_gpu)
        self.max_patch = max_patch
        self.model.config.max_dynamic_patch = self.max_patch
        logging.info(f"Using dynamic batch {self.model.config.max_dynamic_patch}")
        self.tokenizer = AutoTokenizer.from_pretrained(self._path, trust_remote_code=True, use_fast=False)
        # InternLM2 tokenizer sometimes has pad_token_id=None; batched generation needs it
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def score(self, seq_of_frames: list[np.ndarray], prompts: list[str]) -> list[YesNoScore]:
        """Return one ``YesNoScore`` (greedy token, P(Yes)/(P(Yes)+P(No)), Yes+No mass) per prompt."""
        raise NotImplementedError

    def batch_detect(
        self,
        seq_of_frames: list[np.ndarray],
        scene_descriptions: list[str],
        threshold: float,
    ) -> list[DetectedObject]:
        """Detect every proposition in one window of frames."""
        prompts = [detection_prompt(prop) for prop in scene_descriptions]
        scored = self.score(seq_of_frames, prompts)

        detected_objects = []
        for prop, (response, confidence, mass) in zip(scene_descriptions, scored):
            probability = calibrate_sigmoid(confidence, false_threshold=threshold)
            detected_objects.append(
                DetectedObject(
                    name=prop,
                    # as upstream: detected when the greedy answer token says yes
                    is_detected="yes" in response.lower(),
                    confidence=round(confidence, 3),
                    probability=round(probability, 3),
                    yes_no_mass=round(mass, 4),
                )
            )
        return detected_objects

    def clear_gpu_memory(self) -> None:
        """Clear CUDA cache and run garbage collection to free GPU memory."""
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.ipc_collect()
        gc.collect()

    def move_tensors_to_gpu(self, module: torch.nn.Module) -> None:
        """Move all tensors in the module to GPU if they are on the CPU."""
        for name, tensor in module.named_buffers():
            if isinstance(tensor, torch.Tensor) and tensor.device.type == "cpu":
                module.register_buffer(name, tensor.cuda(self.device), persistent=False)
        for _, param in module.named_parameters():
            if param.device.type == "cpu":
                param.data = param.data.cuda(self.device)


# ---------------------------------------------------------------- preprocessing

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size: int) -> T.Compose:
    """Builds a transformation pipeline for the given input size."""
    return T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images


def load_video_from_seq_of_frames(
    seq_of_frames: list[np.ndarray],
    input_size=448,
    max_num=1,
    device="cuda",
    dtype=torch.bfloat16,
):
    pixel_values_list, num_patches_list = [], []
    transform = build_transform(input_size=input_size)
    for img in seq_of_frames:
        img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = torch.stack([transform(tile) for tile in img]).to(dtype=dtype, device=device)
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)
    return torch.cat(pixel_values_list, dim=0), num_patches_list


_NUM_LAYERS = {
    "InternVL2-1B": 24,
    "InternVL2-2B": 24,
    "InternVL2-4B": 32,
    "InternVL2-8B": 32,
    "InternVL2-26B": 48,
    "InternVL2-40B": 60,
    "InternVL2-Llama3-76B": 80,
}


def assign_device_map(model_name, manual_gpu_id=0):
    num_layers = _NUM_LAYERS[model_name]
    device_map = {f"language_model.model.layers.{layer_idx}": manual_gpu_id for layer_idx in range(num_layers)}
    for key in [
        "vision_model",
        "mlp1",
        "language_model.model.tok_embeddings",
        "language_model.model.embed_tokens",
        "language_model.output",
        "language_model.model.norm",
        "language_model.lm_head",
    ]:
        device_map[key] = manual_gpu_id
    logging.info(f"Device is {manual_gpu_id}, model will be loaded there")
    return device_map


def split_model(model_name):
    device_map = {}
    world_size = torch.cuda.device_count()
    num_layers = _NUM_LAYERS[model_name]
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for _ in range(num_layer):
            device_map[f"language_model.model.layers.{layer_cnt}"] = i
            layer_cnt += 1
    for key in [
        "vision_model",
        "mlp1",
        "language_model.model.tok_embeddings",
        "language_model.model.embed_tokens",
        "language_model.output",
        "language_model.model.norm",
        "language_model.lm_head",
    ]:
        device_map[key] = 0
    device_map[f"language_model.model.layers.{num_layers - 1}"] = 0
    return device_map
