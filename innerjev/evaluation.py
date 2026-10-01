"""Capability-equal validation scoring; no test inputs are used here."""
import collections
import math


def evaluate_predictions(rows, predictions):
    lookup = predictions if isinstance(predictions, dict) else {r['id']: r for r in predictions}
    if {r['id'] for r in rows} != set(lookup):
        raise ValueError('Prediction and validation question IDs differ.')
    grouped = collections.defaultdict(list)
    capabilities = {}
    for row in rows:
        probs = lookup[row['id']]['probs']
        target = row['gold_target']
        if len(probs) != len(target):
            raise ValueError('Probability/target lengths differ.')
        nll = -sum(y*math.log(max(p, 1e-30)) for y, p in zip(target, probs))
        if row['type'] == 'score':
            pred = sum(i*p for i, p in enumerate(probs))
            gold = row.get('gold_value')
            if gold is None:
                gold = sum(i*y for i, y in enumerate(target))
            metric, value = 'score_nmae', abs(pred-gold)/(len(probs)-1)
        elif row['type'] == 'noul' and max(target) < 1-1e-6:
            index = next(i for i, option in enumerate(row['options']) if option['key'] in ('yes', 'true'))
            metric, value = 'soft_noul_mae', abs(probs[index]-target[index])
        else:
            metric = 'accuracy'
            value = int(max(range(len(probs)), key=probs.__getitem__)
                        == max(range(len(target)), key=target.__getitem__))
        source = row.get('evaluation_source', row['source'])
        capabilities[source] = row['capability']
        grouped[source].append((metric, value, nll))
    sources = {}
    for source, records in grouped.items():
        by_metric = collections.defaultdict(list)
        for metric, value, _ in records:
            by_metric[metric].append(value)
        values = {m: sum(v)/len(v) for m, v in by_metric.items()}
        score = sum(v if m == 'accuracy' else 1-v for m, v in values.items())/len(values)
        sources[source] = dict(n=len(records), score=100*score,
                               nll=sum(r[2] for r in records)/len(records), **values)
    groups = collections.defaultdict(list)
    for source, value in sources.items():
        groups[capabilities[source]].append(value)
    summary = {group: dict(score=sum(r['score'] for r in values)/len(values),
                            nll=sum(r['nll'] for r in values)/len(values),
                            n=sum(r['n'] for r in values)) for group, values in groups.items()}
    return dict(n=len(rows), score=sum(r['score'] for r in summary.values())/len(summary),
                nll=sum(r['nll'] for r in summary.values())/len(summary), sources=sources,
                capabilities=summary)
