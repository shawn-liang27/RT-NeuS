"""RT-NeuS detector: prefill a window's frames once, answer every proposition from the KV cache."""

import copy

import numpy as np
import torch
from transformers.cache_utils import DynamicCache

from neus_core.vlm.internvl import InternVLDetector, load_video_from_seq_of_frames
from neus_core.vlm.yes_no import YesNoScore, yes_no_confidence

IMG_START_TOKEN = "<img>"
IMG_END_TOKEN = "</img>"
IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"


class KVCacheInternVL(InternVLDetector):
    def score(self, seq_of_frames: list[np.ndarray], prompts: list[str]) -> list[YesNoScore]:
        """Prefill the visual prefix once and reuse its KV cache for every question."""
        prefix = self.prefill_visual_prefix(seq_of_frames)
        scored = [self.answer_with_cache(prefix, prompt) for prompt in prompts]
        # Free the cache before returning so the next call starts clean
        del prefix
        return scored

    def _build_visual_prefix_text(self, num_patches_list):
        """Build the prompt prefix: system + user role open + all frame image tokens.

        Stops BEFORE the question, so this text is question-independent.
        Returns (prefix_text, template). The template is returned so the per-question
        suffix can be assembled consistently (sep, assistant role, etc).
        """
        template = copy.deepcopy(self.model.conv_template)
        template.system_message = self.model.system_message

        # For InternLM2 chat this is: "<|im_start|>system\n{sys}<|im_end|>\n<|im_start|>user\n"
        prefix = template.system_template.format(system_message=template.system_message)
        prefix += template.sep + template.roles[0] + "\n"

        for i, num_patches in enumerate(num_patches_list):
            context_tokens = IMG_CONTEXT_TOKEN * self.model.num_image_token * num_patches
            prefix += f"Frame{i + 1}: " + IMG_START_TOKEN + context_tokens + IMG_END_TOKEN + "\n"
        return prefix, template

    @torch.no_grad()
    def prefill_visual_prefix(self, seq_of_frames):
        """Run ViT + LM prefill over the visual prefix. Returns cache + metadata.

        This is the expensive, question-independent work. Call once per window of
        frames, then call `answer_with_cache` repeatedly for each question.
        """
        pixel_values, num_patches_list = load_video_from_seq_of_frames(
            seq_of_frames=seq_of_frames, device=self.device, max_num=self.max_patch
        )

        img_context_token_id = self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.model.img_context_token_id = img_context_token_id

        vit_embeds = self.model.extract_feature(pixel_values)        # [P, T_img, H]
        vit_embeds = vit_embeds.reshape(-1, vit_embeds.shape[-1])    # [P*T_img, H]

        prefix_text, template = self._build_visual_prefix_text(num_patches_list)
        enc = self.tokenizer(prefix_text, return_tensors="pt", add_special_tokens=False)
        input_ids = enc.input_ids.to(self.device)
        attention_mask = enc.attention_mask.to(self.device)

        # Embed and splice vision features into IMG_CONTEXT slots
        input_embeds = self.model.language_model.get_input_embeddings()(input_ids).clone()
        selected = input_ids == img_context_token_id
        assert selected.sum() == vit_embeds.shape[0], (
            f"IMG_CONTEXT count {selected.sum().item()} != vision feature count {vit_embeds.shape[0]}"
        )
        input_embeds[selected] = vit_embeds.to(input_embeds.dtype)

        # Prefill the LM — pass None so the model creates the cache internally
        output = self.model.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            past_key_values=None,
            use_cache=True,
            return_dict=True,
        )
        cache = output.past_key_values

        # InternLM2 only accepts tuple-of-tuples cache, not DynamicCache
        if isinstance(cache, DynamicCache):
            cache = tuple((k, v) for k, v in zip(cache.key_cache, cache.value_cache))

        return {
            "cache": cache,
            "attention_mask": attention_mask,   # shape [1, prefix_len]
            "prefix_len": input_ids.shape[1],
            "template": template,
        }

    @staticmethod
    def _clone_cache(cache):
        """Deep-clone a tuple-of-tuples KV cache so per-question generation
        does not contaminate the shared prefix cache."""
        return tuple((k.clone(), v.clone()) for k, v in cache)

    @torch.no_grad()
    def answer_with_cache(self, prefix, question):
        """Answer a yes/no question reusing the prefilled visual prefix KV cache.

        Uses a single forward pass instead of generate() — generate()'s
        prepare_inputs_for_generation assumes input_ids starts from position 0,
        which conflicts with our pre-filled cache.

        Returns:
            YesNoScore(greedy token, P(Yes) / (P(Yes) + P(No)), Yes+No mass)
        """
        template = prefix["template"]
        prefix_len = prefix["prefix_len"]

        suffix = question + template.sep + template.roles[1] + "\n"
        q_ids = self.tokenizer(suffix, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)
        q_len = q_ids.shape[1]

        cache = self._clone_cache(prefix["cache"])

        # Attention mask covers prefix (in cache) + question tokens
        full_attn = torch.cat([prefix["attention_mask"], torch.ones_like(q_ids)], dim=1)

        # Position IDs for question tokens continue from where prefix ended
        position_ids = torch.arange(prefix_len, prefix_len + q_len, device=self.device).unsqueeze(0)

        output = self.model.language_model(
            input_ids=q_ids,
            attention_mask=full_attn,
            past_key_values=cache,
            position_ids=position_ids,
            use_cache=False,
            return_dict=True,
        )

        # Logits at the last position = prediction for the first answer token,
        # scored as P(Yes) / (P(Yes) + P(No))
        return yes_no_confidence(output.logits[0, -1, :], self.tokenizer)
