import torch
import torch.nn.functional as F
from flash_attn.flash_attn_interface import _flash_attn_forward
from flash_attn import flash_attn_func
import torch
import torch.distributed as dist
import os

"""
All-reduce Attention just different communication from Blockwise attention,
only make sense for distributed message passing.

```bash
torchrun --nproc_per_node=3 allreduce-fa2.py
```

Output,

```
0 torch.Size([1, 16, 20, 128])
4 torch.Size([1, 16, 20, 128])
1 torch.Size([1, 16, 20, 128])
2 torch.Size([1, 16, 20, 128])
3 torch.Size([1, 16, 20, 128])
```

To verify the integrity of Blockwise attention, check out prefill-fa2.ipynb
"""

if __name__ == "__main__":
    world_size = int(os.environ['LOCAL_WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    dist.init_process_group(backend='nccl')
    device = torch.device(f'cuda:{local_rank}')

    batch_size = 1
    head_num = 16
    dim = 128
    seq_len = 100 // world_size

    q = torch.randn(batch_size, seq_len, head_num, dim, device=device, dtype=torch.bfloat16)
    k = torch.randn(batch_size, seq_len, head_num, dim, device=device, dtype=torch.bfloat16)
    v = torch.randn(batch_size, seq_len, head_num, dim, device=device, dtype=torch.bfloat16)

    softmax_scale = q.shape[-1] ** (-0.5)

    outputs = _flash_attn_forward(
        q, 
        k, 
        v, 
        dropout_p=0, 
        softmax_scale=softmax_scale, 
        causal=False,
        window_size_left=-1,
        window_size_right=-1,
        softcap=0.0,
        alibi_slopes=None,
        return_softmax=False,
    )
    block_out, block_lse, _, _ = outputs
    block_lse = block_lse.transpose(1, 2)
    
    global_max_lse = block_lse.clone()
    dist.all_reduce(global_max_lse.contiguous(), op=dist.ReduceOp.MAX)
    w = torch.exp(block_lse - global_max_lse)
    weighted_out = block_out * w[..., None]
    dist.all_reduce(weighted_out.contiguous(), op=dist.ReduceOp.SUM)
    dist.all_reduce(w.contiguous(), op=dist.ReduceOp.SUM)
    attn_output = weighted_out / w[..., None]
    lse = torch.log(w) + global_max_lse
    
    print(local_rank, attn_output.shape)