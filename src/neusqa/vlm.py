"""NeuS-QA detector: one forward pass per proposition per window.

This is the upstream NeuS-QA detection scheme (each proposition asked on its
own) running on HuggingFace InternVL instead of a vLLM server.
"""

import copy

import numpy as np
import torch

from neus_core.vlm.internvl import InternVLDetector, load_video_from_seq_of_frames
from neus_core.vlm.yes_no import YesNoScore, yes_no_confidence


class SequentialInternVL(InternVLDetector):
    def score(self, seq_of_frames: list[np.ndarray], prompts: list[str]) -> list[YesNoScore]:
        pixel_values, num_patches_list = load_video_from_seq_of_frames(
            seq_of_frames=seq_of_frames, device=self.device, max_num=1
        )
        video_prefix = "".join(f"Frame{i + 1}: <image>\n" for i in range(len(num_patches_list)))
        return [self._ask(pixel_values, num_patches_list, video_prefix + prompt) for prompt in prompts]

    @torch.no_grad()
    def _ask(self, pixel_values, num_patches_list, question):
        """Answer one yes/no question with a single generated token."""
        if hasattr(self.model, "vision_model"):
            target_device = self.model.vision_model.embeddings.patch_embedding.weight.device
        else:
            target_device = self.model.device

        pixel_values = pixel_values.to(target_device).to(self.model.dtype).contiguous()
        if torch.isnan(pixel_values).any() or torch.isinf(pixel_values).any():
            print("CRITICAL: Found NaNs/Infs in video! Scrubbing data...", flush=True)
            pixel_values = torch.nan_to_num(pixel_values, nan=0.0, posinf=1.0, neginf=-1.0)

        img_context_token = "<IMG_CONTEXT>"
        self.model.img_context_token_id = self.tokenizer.convert_tokens_to_ids(img_context_token)

        template = copy.deepcopy(self.model.conv_template)
        template.system_message = self.model.system_message
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()
        for num_patches in num_patches_list:
            image_tokens = "<img>" + img_context_token * self.model.num_image_token * num_patches + "</img>"
            query = query.replace("<image>", image_tokens, 1)

        model_inputs = self.tokenizer(query, return_tensors="pt")
        generation_output = self.model.generate(
            pixel_values=pixel_values,
            input_ids=model_inputs["input_ids"].to(target_device),
            attention_mask=model_inputs["attention_mask"].to(target_device),
            max_new_tokens=1,
            do_sample=False,
            eos_token_id=self.tokenizer.convert_tokens_to_ids(template.sep),
            return_dict_in_generate=True,
            output_logits=True,
        )
        # Score from the first answer token: P(Yes) / (P(Yes) + P(No))
        return yes_no_confidence(generation_output.logits[0][0], self.tokenizer)
