import torch
import torch.nn as nn

class PromptLearnerForBERT(nn.Module):
    def __init__(self, sample_id_list, embedding_dim=768, n_prompt=20, position="end"):
        super().__init__()
        assert position in {"front", "middle", "end"}, "position must be front/middle/end"
        self.position = position
        self.n_prompt = n_prompt
        self.embedding_dim = embedding_dim

        self.sample_id_to_index = {sid: i for i, sid in enumerate(sample_id_list)}
        self.prompt_embeddings = nn.Parameter(
            torch.randn(len(sample_id_list), n_prompt, embedding_dim)
        )
        nn.init.normal_(self.prompt_embeddings, std=0.02)

    def forward(self, sample_id_tensor, input_embeds, attention_mask):
        B, L, D = input_embeds.shape
        idx = torch.tensor(
            [self.sample_id_to_index[int(sid)] for sid in sample_id_tensor],
            dtype=torch.long, device=self.prompt_embeddings.device
        ) 
        prompts = self.prompt_embeddings[idx] 

        inputs_embeds_new = input_embeds.clone()
        attention_mask_new = attention_mask.clone()

        for b in range(B):
            pad_positions = (attention_mask[b] == 0).nonzero(as_tuple=False).squeeze(-1)  # (K,)
            K = pad_positions.numel()
            if K == 0:
                continue
            use_k = min(K, self.n_prompt)
            if use_k < K:
                pad_positions = pad_positions[:use_k]
                K = use_k

            inputs_embeds_new[b, pad_positions, :] = prompts[b, :K, :]
            attention_mask_new[b, pad_positions] = 1

        return inputs_embeds_new, attention_mask_new