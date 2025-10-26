import torch, torch.nn.functional as F

def ce_full_softmax(user_vec: torch.Tensor, item_table: torch.Tensor, targets: torch.Tensor):
    """
    user_vec: [B, d]
    item_table: [|I|+1, d]，第0行是PAD
    targets: 真实 item id ∈ [1..|I|]
    """
    logits = user_vec @ item_table[1:].T    # [B, |I|]
    loss = F.cross_entropy(logits, targets - 1)
    return loss, logits
