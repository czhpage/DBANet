import torch
import torch.nn.functional as F
from einops import rearrange

def calculate_confidence(logits):
    probabilities = F.softmax(logits, dim=1)  
    confidence, _ = torch.max(probabilities, dim=1)
    return confidence


def merge_patches(logits_A, logits_B, patch_size):
    patch_D, patch_H, patch_W = patch_size
    conf_A = calculate_confidence(logits_A)
    conf_B = calculate_confidence(logits_B)

    patches_conf_A = rearrange(conf_A, 'b (d pd) (h ph) (w pw) -> b d h w pd ph pw', 
                               pd=patch_D, ph=patch_H, pw=patch_W)
    patches_conf_B = rearrange(conf_B, 'b (d pd) (h ph) (w pw) -> b d h w pd ph pw', 
                               pd=patch_D, ph=patch_H, pw=patch_W)
    
    patches_logits_A = rearrange(logits_A, 'b c (d pd) (h ph) (w pw) -> b c d h w pd ph pw', 
                                 pd=patch_D, ph=patch_H, pw=patch_W)
    patches_logits_B = rearrange(logits_B, 'b c (d pd) (h ph) (w pw) -> b c d h w pd ph pw', 
                                 pd=patch_D, ph=patch_H, pw=patch_W)

    patch_conf_A = patches_conf_A.mean(dim=(-1, -2, -3))
    patch_conf_B = patches_conf_B.mean(dim=(-1, -2, -3))

    merge_mask = patch_conf_A > patch_conf_B
    merge_mask = merge_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    merged_patches = torch.where(merge_mask, patches_logits_A, patches_logits_B)
    merged_logits = rearrange(merged_patches, 'b c d h w pd ph pw -> b c (d pd) (h ph) (w pw)')

    return merged_logits


