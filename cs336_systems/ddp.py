import torch
import torch.nn as nn
import torch.distributed as dist
from cs336_basics.lang_model import cross_entropy, AdamW, TransformerLM

class NaiveDDP(nn.Module):
    def __init__(self, module: nn.Module, flat_gradients: bool = False):
        super().__init__()
        self.module = module
        self.flat_gradients = flat_gradients

        # broadcast parameters from rank 0 to all ranks
        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def after_backward(self):
        world_size = dist.get_world_size()

        if self.flat_gradients:
            self._after_backward_flat(world_size)
        else:
            self._after_backward_individual(world_size)

    def _after_backward_individual(self, world_size):
        """Original: one async all_reduce per parameter."""
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

    def _after_backward_flat(self, world_size):
        """Batched: concatenate all gradients into one tensor, single all_reduce."""
        # collect all gradients that exist
        grads = [p.grad.data for p in self.module.parameters() if p.grad is not None]
        params_with_grads = [p for p in self.module.parameters() if p.grad is not None]

        if not grads:
            return

        # flatten all gradients into a single 1D tensor
        flat = torch._utils._flatten_dense_tensors(grads)

        # single all_reduce on the flat tensor
        dist.all_reduce(flat, op=dist.ReduceOp.SUM, async_op=False)
        flat /= world_size

        # unflatten and copy back into each parameter's grad
        unflat = torch._utils._unflatten_dense_tensors(flat, grads)
        for param, new_grad in zip(params_with_grads, unflat):
            param.grad.data.copy_(new_grad)

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

class DDP(nn.Module):
    """
    Distributed Data Parallel implementation that overlaps gradient
    communication with the backward pass using autograd hooks.
    """

    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        self.handles = []  # async all_reduce handles

        # broadcast parameters from rank 0 to all ranks
        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)

        # register hook on each parameter to fire when its
        # gradient is ready during backward pass
        self.world_size = dist.get_world_size()
        for param in self.module.parameters():
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(
                    self._make_hook(param)
                )

    def _make_hook(self, param):
        """
        Returns a hook that fires async all_reduce as soon as
        this parameter's gradient is ready during backward pass.
        """
        def hook(p):
            # immediately fire async all_reduce for this gradient
            handle = dist.all_reduce(
                p.grad.data,
                op=dist.ReduceOp.SUM,
                async_op=True,
            )
            self.handles.append((handle, p))
        return hook

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self):
        """
        Wait for all async all_reduce operations to complete
        then divide by world_size to get averaged gradients.
        Call this after loss.backward() and before optimizer.step().
        """
        for handle, param in self.handles:
            handle.wait()
            param.grad.data /= self.world_size
        self.handles.clear()