import torch
import torch.nn as nn
import torch.distributed as dist
from cs336_basics.lang_model import cross_entropy, AdamW, TransformerLM

class NaiveDDP(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

        # broadcast parameters from rank 0 to all ranks
        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def after_backward(self):
        """
        All-reduce gradients across all ranks after backward pass.
        Uses async all_reduce for each parameter individually,
        then waits for all to complete before dividing by world_size.
        """
        world_size = dist.get_world_size()
        handles = []

        # fire all async all_reduce operations
        for param in self.module.parameters():
            if param.grad is not None:
                handle = dist.all_reduce(
                    param.grad.data,
                    op=dist.ReduceOp.SUM,
                    async_op=True,
                )
                handles.append((handle, param))

        # wait for all to complete then average
        for handle, param in handles:
            handle.wait()
            param.grad.data /= world_size

def ddp_on_after_backward(ddp_model: NaiveDDP):
    """
    Adapter function that calls after_backward on the DDP model.
    Called after loss.backward() and before optimizer.step().
    """
    ddp_model.after_backward()

def train_step(ddp_model, optimizer, inputs, targets):
    # 1. forward pass on local shard of batch
    optimizer.zero_grad()
    logits = ddp_model(inputs)
    loss = cross_entropy(logits, targets)

    # 2. backward pass — computes local gradients
    loss.backward()

    # 3. all-reduce gradients across all ranks
    ddp_model.after_backward()

    # 4. optimizer step — identical on all ranks since gradients are averaged
    optimizer.step()

    return loss.item()

def get_batch(data, batch_size, rank, world_size, device):
    """
    Get a shard of the batch for this rank.
    Each rank gets batch_size // world_size examples.
    """
    shard_size = batch_size // world_size
    start = rank * shard_size
    end = start + shard_size
    return data[start:end].to(device), data[start+1:end+1].to(device)

def run_ddp_training(rank, world_size, data, batch_size, num_steps, lr=3e-4, backend="nccl"):
    # setup
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(rank)

    # build model and wrap in DDP
    model = TransformerLM(...).to(device)
    ddp_model = NaiveDDP(model)
    optimizer = AdamW(ddp_model.parameters(), lr)

    for step in range(num_steps):
        # shard batch — each rank gets batch_size/world_size examples
        inputs, targets = get_batch(data, batch_size, rank, world_size, device)

        # forward + backward
        optimizer.zero_grad()
        logits = ddp_model(inputs)
        loss = cross_entropy(logits, targets)
        loss.backward()

        # all-reduce gradients
        ddp_model.after_backward()

        # optimizer step — stays in sync on all ranks
        optimizer.step()

    dist.destroy_process_group()