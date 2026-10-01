"""Deterministic length buckets, equal distributed microbatches, and LR schedule."""
import hashlib
import json
import math
import random


def batch_plan(rows, lengths, seed, epochs=1):
    result = []
    for epoch in range(epochs):
        def ordering(index):
            row = rows[index]
            key = row.get('ordering_key') if epoch == 0 else None
            if key is None:
                key = hashlib.sha256(json.dumps([seed, epoch, row['id']], sort_keys=True).encode()).hexdigest()
            return lengths[index], key
        order = sorted(range(len(rows)), key=ordering)
        batches = [order[i:i+64] for i in range(0, len(order), 64)]
        random.Random(seed + epoch).shuffle(batches)
        result.extend(batches)
    return result


def split_batch(indices, lengths, world, micro, budget=8192):
    padded = indices + [indices[-1]] * (64-len(indices))
    micro = min(micro, 64 // world)
    while micro > 1 and micro * max(lengths[i] for i in indices) > budget:
        micro //= 2
    if max(lengths[i] for i in indices) > budget or 64 % (world*micro):
        raise ValueError('Incompatible microbatch, world size, or token budget.')
    return [[[(j, padded[j]) for j in range(offset + rank*micro, offset + (rank+1)*micro)]
             for offset in range(0, 64, world*micro)] for rank in range(world)]


def lr_factor(step, total):
    index = step-1
    warmup = max(1, math.ceil(.03*total))
    return step/warmup if index < warmup else .5*(1+math.cos(math.pi*(index-warmup)/max(1, total-warmup)))


def select_checkpoint(entries):
    top = max(entry['score'] for entry in entries)
    return min((entry for entry in entries if top-entry['score'] < .1),
               key=lambda entry: (entry['nll'], entry['step']))
