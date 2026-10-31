"""Same-layout FSDP recovery at optimizer-step boundaries (use_orig_params=True).

These local shards are restart artifacts, not release/model checkpoints.
Resume requires the same world size, parameter layout, recipe and data order.
"""

import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch
import torch.distributed as dist


def cpu_copy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_copy(item) for item in value)
    return value


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save_recovery(model, optimizer, root, completed, binding):
    """Write CPU copies of local shards, then publish a complete generation."""
    started = time.perf_counter()
    rank, world = dist.get_rank(), dist.get_world_size()
    folder = Path(root) / f'step_{completed:09d}'
    if rank == 0:
        folder.mkdir(parents=True, exist_ok=True)
        if (folder / 'complete.json').exists():
            raise FileExistsError(folder)
    dist.barrier()
    payload = {
        'binding': binding, 'rank': rank, 'world_size': world,
        'completed_steps': completed,
        'parameters': {k: cpu_copy(v) for k, v in model.named_parameters()},
        'buffers': {k: cpu_copy(v) for k, v in model.named_buffers()},
        'optimizer': cpu_copy(optimizer.state_dict()),
        'torch_rng': torch.get_rng_state(),
        'cuda_rng': torch.cuda.get_rng_state(),
        'python_rng': random.getstate(), 'numpy_rng': np.random.get_state(),
    }
    target = folder / f'rank_{rank}.pt'
    temporary = target.with_suffix('.pt.partial')
    with temporary.open('wb') as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, target)
    del payload
    row = {'rank': rank, 'file': target.name, 'bytes': target.stat().st_size,
           'sha256': digest(target)}
    shards = [None] * world
    dist.all_gather_object(shards, row)
    if rank == 0:
        manifest = {'completed_steps': completed, 'world_size': world,
                    'binding': binding, 'shards': shards,
                    'save_seconds': time.perf_counter() - started}
        temporary = folder / 'complete.json.partial'
        with temporary.open('w') as stream:
            json.dump(manifest, stream, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, folder / 'complete.json')
        # Only committed generations count; an interrupted write never replaces one.
        complete = sorted(Path(root).glob('step_*/complete.json'))
        for old in complete[:-2]:
            shutil.rmtree(old.parent)
        print(json.dumps({'recovery_saved': str(folder), **manifest}), flush=True)
    dist.barrier()
    return time.perf_counter() - started


def load_recovery(model, optimizer, folder, binding):
    folder = Path(folder)
    rank, world = dist.get_rank(), dist.get_world_size()
    manifest = json.loads((folder / 'complete.json').read_text())
    if manifest['binding'] != binding or manifest['world_size'] != world:
        raise RuntimeError('Recovery recipe/data/code/world size mismatch')
    shard = manifest['shards'][rank]
    path = folder / shard['file']
    if path.stat().st_size != shard['bytes'] or digest(path) != shard['sha256']:
        raise RuntimeError('Recovery shard is incomplete or corrupted')
    state = torch.load(path, map_location='cpu', weights_only=False)
    assert state['rank'] == rank and state['binding'] == binding
    assert state['completed_steps'] == manifest['completed_steps']
    with torch.no_grad():
        for key, current in [('parameters', dict(model.named_parameters())),
                             ('buffers', dict(model.named_buffers()))]:
            saved = state[key]
            if saved.keys() != current.keys():
                raise RuntimeError('Recovery parameter names mismatch')
            for name, value in current.items():
                if value.shape != saved[name].shape or value.dtype != saved[name].dtype:
                    raise RuntimeError(f'Recovery local layout mismatch: {name}')
                value.copy_(saved[name])
    optimizer.load_state_dict(state['optimizer'])
    torch.set_rng_state(state['torch_rng'])
    torch.cuda.set_rng_state(state['cuda_rng'])
    random.setstate(state['python_rng'])
    np.random.set_state(state['numpy_rng'])
    # Initialization and optimizer restoration allocate in a different order
    # from the first training step. Release their unused allocator blocks once.
    del state
    torch.cuda.empty_cache()
    dist.barrier()
    return int(manifest['completed_steps'])
