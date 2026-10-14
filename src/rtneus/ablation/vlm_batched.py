"""Ablation detector: all propositions of a window in one batched forward pass.

RT-NeuS batching without KV-cache reuse: the visual features are computed
once and repeated for every question in the batch.
"""

import copy

import numpy as np
import torch

from neus_core.vlm.internvl import InternVLDetector, load_video_from_seq_of_frames
from neus_core.vlm.yes_no import YesNoScore, yes_no_confidence


class BatchedInternVL(InternVLDetector):
    @torch.no_grad()
    def score(self, seq_of_frames: list[np.ndarray], prompts: list[str]) -> list[YesNoScore]:
        pixel_values, num_patches_list = load_video_from_seq_of_frames(
            seq_of_frames=seq_of_frames, device=self.device, max_num=self.max_patch
        )
        video_prefix = "".join(f"Frame{i + 1}: <image>\n" for i in range(len(num_patches_list)))

        img_context_token = "<IMG_CONTEXT>"
        self.model.img_context_token_id = self.tokenizer.convert_tokens_to_ids(img_context_token)

        queries = []
        for prompt in prompts:
            template = copy.deepcopy(self.model.conv_template)
            template.system_message = self.model.system_message
            template.append_message(template.roles[0], video_prefix + prompt)
            template.append_message(template.roles[1], None)
            query = template.get_prompt()
            for num_patches in num_patches_list:
                image_tokens = "<img>" + img_context_token * self.model.num_image_token * num_patches + "</img>"
                query = query.replace("<image>", image_tokens, 1)
            queries.append(query)

        self.tokenizer.padding_side = "left"
        model_inputs = self.tokenizer(queries, return_tensors="pt", padding=True, add_special_tokens=False)

        # Visual features once, repeated for every question in the batch
        hidden_dim = self.model.language_model.config.hidden_size
        visual_block = self.model.extract_feature(pixel_values).view(1, -1, hidden_dim)
        expanded_features = visual_block.expand(len(prompts), -1, -1).contiguous()

        generation_output = self.model.generate(
            pixel_values=pixel_values[0:1],
            visual_features=expanded_features,
            input_ids=model_inputs["input_ids"].to(self.device),
            attention_mask=model_inputs["attention_mask"].to(self.device),
            max_new_tokens=1,
            min_new_tokens=1,
            do_sample=False,
            eos_token_id=self.tokenizer.convert_tokens_to_ids(template.sep.strip()),
            pad_token_id=self.tokenizer.pad_token_id,
            return_dict_in_generate=True,
            output_logits=True,
        )
        # Score from the first answer token: P(Yes) / (P(Yes) + P(No))
        first_logits = generation_output.logits[0]
        return [yes_no_confidence(first_logits[b], self.tokenizer) for b in range(first_logits.shape[0])]
