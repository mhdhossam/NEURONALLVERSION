#!/usr/bin/env python3
"""Test whether error-only exemplar feedback generalizes to unseen examples.

Uses sklearn's handwritten-digits dataset and the exact metadata-selection guard
extracted from Untitled36.ipynb. It compares error-driven updates with equal-count,
per-class random exemplar growth on a frozen known/unknown audit set.
"""
import argparse, ast, json, os, statistics, time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = Path(os.environ.get('NEURON_NOTEBOOK_PATH', ROOT / 'Untitled36.ipynb'))
KNOWN = set(range(8))
UNKNOWN = {8, 9}
SEEDS = list(range(2026, 2036))


def load_guard():
    nb = json.loads(NOTEBOOK.read_text(encoding='utf-8'))
    source = next(
        ''.join(c.get('source', [])) for c in nb['cells']
        if '# Metadata-selection guard' in ''.join(c.get('source', []))
        and 'def find_best_in_candidates' in ''.join(c.get('source', []))
    )
    tree = ast.parse(source)
    assigned = {
        'FEATURES_BY_NEURON', 'NEURON_METADATA_MIN_SIMILARITY',
        'NEURON_METADATA_MIN_MARGIN', 'NEURON_METADATA_TOP_EXEMPLARS',
        'LAST_METADATA_DECISION',
    }
    funcs = {'_normalise_embedding', 'find_best_in_candidates'}
    nodes = []
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(t, ast.Name) and t.id in assigned for t in targets):
                nodes.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in funcs:
            nodes.append(node)
    namespace = {'np': np}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(NOTEBOOK), 'exec'), namespace)
    assert assigned | funcs <= set(namespace)
    return namespace


def prediction(guard, brain, candidates, embedding):
    node_id, score = guard['find_best_in_candidates'](brain, embedding, candidates)
    diagnostic = dict(guard['LAST_METADATA_DECISION'])
    label = brain.neurons[node_id].soma.metadata['label'] if node_id is not None else None
    reason = diagnostic.get('reason', 'unknown_rejection') if label is None else 'accepted'
    return label, float(score), diagnostic, reason


def known_metrics(predictions, truth):
    accepted = np.array([p is not None for p in predictions], dtype=bool)
    correct = np.array([p == str(int(y)) for p, y in zip(predictions, truth)], dtype=bool)
    return {
        'n': len(truth),
        'accuracy': round(float(correct.mean()), 4),
        'coverage': round(float(accepted.mean()), 4),
        'selective_accuracy': round(float((correct & accepted).sum() / max(1, accepted.sum())), 4),
        'correct': int(correct.sum()),
        'rejected': int((~accepted).sum()),
        'wrong_accepted': int((accepted & ~correct).sum()),
    }


def unknown_metrics(predictions):
    rejected = sum(p is None for p in predictions)
    n = len(predictions)
    return {
        'n': n, 'rejected': int(rejected),
        'rejection_rate': round(rejected / max(1, n), 4),
        'false_accepts': int(n - rejected),
        'false_accept_rate': round((n - rejected) / max(1, n), 4),
    }


def run_seed(features, labels, seed, guard):
    train_ids, audit_ids = train_test_split(
        np.arange(len(labels)), test_size=0.25, stratify=labels, random_state=seed
    )
    known_train = train_ids[np.isin(labels[train_ids], sorted(KNOWN))]
    known_audit = audit_ids[np.isin(labels[audit_ids], sorted(KNOWN))]
    unknown_audit = audit_ids[np.isin(labels[audit_ids], sorted(UNKNOWN))]
    rng = np.random.default_rng(seed)
    base_ids, feedback_ids = [], []
    for label in sorted(KNOWN):
        rows = known_train[labels[known_train] == label].copy()
        rng.shuffle(rows)
        split = int(round(len(rows) * 0.60))
        base_ids.extend(rows[:split].tolist())
        feedback_ids.extend(rows[split:].tolist())
    base_ids = np.asarray(base_ids, dtype=int)
    feedback_ids = np.asarray(feedback_ids, dtype=int)
    rng.shuffle(feedback_ids)

    label_to_node = {label: f'digit_{label}' for label in sorted(KNOWN)}
    brain = SimpleNamespace(neurons={
        node: SimpleNamespace(soma=SimpleNamespace(metadata={'label': str(label)}))
        for label, node in label_to_node.items()
    })
    candidates = set(brain.neurons)
    bank = guard['FEATURES_BY_NEURON']
    bank.clear()
    for label in sorted(KNOWN):
        bank[label_to_node[label]] = [
            features[i].copy() for i in base_ids if labels[i] == label
        ]
    initial_bank = {nid: [x.copy() for x in rows] for nid, rows in bank.items()}

    # Frozen baseline on the untouched audit sets.
    baseline_known = [prediction(guard, brain, candidates, features[i]) for i in known_audit]
    baseline_unknown = [prediction(guard, brain, candidates, features[i]) for i in unknown_audit]
    baseline_known_labels = [p[0] for p in baseline_known]
    baseline_unknown_labels = [p[0] for p in baseline_unknown]
    baseline_known_metrics = known_metrics(baseline_known_labels, labels[known_audit])
    baseline_unknown_metrics = unknown_metrics(baseline_unknown_labels)

    # Feedback is sequential: reveal the true label only after the current prediction.
    bank.clear()
    bank.update({nid: [x.copy() for x in rows] for nid, rows in initial_bank.items()})
    corrected_sample_ids, per_class_update_count, observed_patterns = set(), {k: 0 for k in sorted(KNOWN)}, set()
    correction_log, feedback_predictions = [], []
    for idx in feedback_ids:
        predicted, score, diagnostic, reason = prediction(guard, brain, candidates, features[idx])
        feedback_predictions.append(predicted)
        true_label = str(int(labels[idx]))
        if predicted != true_label:
            pattern = (true_label, predicted if predicted is not None else '<REJECTED>', reason)
            observed_patterns.add(pattern)
            corrected_sample_ids.add(int(idx))
            per_class_update_count[int(labels[idx])] += 1
            normalized = guard['_normalise_embedding'](features[idx])
            if normalized is None:
                raise RuntimeError('Valid digit feature unexpectedly failed normalization')
            bank[label_to_node[int(labels[idx])]].append(normalized.copy())
            replay_prediction, _, _, replay_reason = prediction(guard, brain, candidates, features[idx])
            correction_log.append({
                'true_label': true_label,
                'prediction_before': predicted,
                'reason_before': reason,
                'prediction_on_replay': replay_prediction,
                'reason_on_replay': replay_reason,
                'replay_fixed': replay_prediction == true_label,
            })

    error_known = [prediction(guard, brain, candidates, features[i]) for i in known_audit]
    error_unknown = [prediction(guard, brain, candidates, features[i]) for i in unknown_audit]
    error_known_labels = [p[0] for p in error_known]
    error_unknown_labels = [p[0] for p in error_unknown]
    error_known_metrics = known_metrics(error_known_labels, labels[known_audit])
    error_unknown_metrics = unknown_metrics(error_unknown_labels)

    # Count/class-matched control: append the same number of non-correction feedback examples.
    non_error_pool = {
        label: [int(i) for i in feedback_ids
                if int(labels[i]) == label and int(i) not in corrected_sample_ids]
        for label in sorted(KNOWN)
    }
    control_ids, fallback_classes = [], []
    for label, count in per_class_update_count.items():
        pool = non_error_pool[label]
        rng.shuffle(pool)
        chosen = pool[:count]
        if len(chosen) < count:
            banned = set(base_ids) | corrected_sample_ids | set(chosen)
            fallback = [int(i) for i in known_train
                        if int(labels[i]) == label and int(i) not in banned]
            rng.shuffle(fallback)
            chosen.extend(fallback[:count - len(chosen)])
            if len(chosen) < count:
                fallback_classes.append(label)
        control_ids.extend(chosen)
    bank.clear()
    bank.update({nid: [x.copy() for x in rows] for nid, rows in initial_bank.items()})
    for idx in control_ids:
        bank[label_to_node[int(labels[idx])]].append(
            guard['_normalise_embedding'](features[idx]).copy()
        )
    random_known = [prediction(guard, brain, candidates, features[i]) for i in known_audit]
    random_unknown = [prediction(guard, brain, candidates, features[i]) for i in unknown_audit]
    random_known_labels, random_unknown_labels = [p[0] for p in random_known], [p[0] for p in random_unknown]
    random_known_metrics = known_metrics(random_known_labels, labels[known_audit])
    random_unknown_metrics = unknown_metrics(random_unknown_labels)

    # Same-failure-signature cohort in the never-trained-on known audit set.
    pattern_cases = []
    for position, idx in enumerate(known_audit):
        before, _, _, reason = baseline_known[position]
        truth = str(int(labels[idx]))
        pattern = (truth, before if before is not None else '<REJECTED>', reason)
        if before != truth and pattern in observed_patterns:
            pattern_cases.append({
                'true_label': truth,
                'baseline_prediction': before,
                'failure_reason': reason,
                'after_error_feedback': error_known_labels[position],
                'after_random_control': random_known_labels[position],
                'error_feedback_fixed': error_known_labels[position] == truth,
                'random_control_fixed': random_known_labels[position] == truth,
            })

    baseline_correct = [p == str(int(y)) for p, y in zip(baseline_known_labels, labels[known_audit])]
    error_correct = [p == str(int(y)) for p, y in zip(error_known_labels, labels[known_audit])]
    random_correct = [p == str(int(y)) for p, y in zip(random_known_labels, labels[known_audit])]
    repairs = sum(not a and b for a, b in zip(baseline_correct, error_correct))
    regressions = sum(a and not b for a, b in zip(baseline_correct, error_correct))
    random_repairs = sum(not a and b for a, b in zip(baseline_correct, random_correct))
    random_regressions = sum(a and not b for a, b in zip(baseline_correct, random_correct))
    return {
        'seed': seed,
        'counts': {
            'initial_memory': int(len(base_ids)), 'feedback_stream': int(len(feedback_ids)),
            'known_audit': int(len(known_audit)), 'unknown_audit': int(len(unknown_audit)),
            'correction_events': len(correction_log), 'random_control_added': len(control_ids),
            'correction_counts_by_true_class': {str(k): int(v) for k, v in per_class_update_count.items()},
            'failure_patterns_observed': len(observed_patterns),
            'matching_unseen_audit_cases': len(pattern_cases),
        },
        'known_audit': {
            'frozen_baseline': baseline_known_metrics,
            'after_error_feedback': error_known_metrics,
            'after_count_class_matched_random_examples': random_known_metrics,
            'error_feedback_accuracy_delta_pp': round(100 * (error_known_metrics['accuracy'] - baseline_known_metrics['accuracy']), 2),
            'random_control_accuracy_delta_pp': round(100 * (random_known_metrics['accuracy'] - baseline_known_metrics['accuracy']), 2),
            'error_feedback_repairs': int(repairs), 'error_feedback_regressions': int(regressions),
            'random_control_repairs': int(random_repairs), 'random_control_regressions': int(random_regressions),
        },
        'unknown_audit': {
            'frozen_baseline': baseline_unknown_metrics,
            'after_error_feedback': error_unknown_metrics,
            'after_count_class_matched_random_examples': random_unknown_metrics,
        },
        'same_failure_pattern_generalization': {
            'n_unseen_cases': len(pattern_cases),
            'error_feedback_corrected': sum(x['error_feedback_fixed'] for x in pattern_cases),
            'error_feedback_fix_rate': round(sum(x['error_feedback_fixed'] for x in pattern_cases) / max(1, len(pattern_cases)), 4),
            'random_control_corrected': sum(x['random_control_fixed'] for x in pattern_cases),
            'random_control_fix_rate': round(sum(x['random_control_fixed'] for x in pattern_cases) / max(1, len(pattern_cases)), 4),
            'cases': pattern_cases[:50],
        },
        'correction_log': correction_log[:100],
        'random_control_fallback_classes': fallback_classes,
    }


def aggregate(values):
    return {
        'mean': round(statistics.mean(values), 4),
        'std_population': round(statistics.pstdev(values), 4),
        'per_seed': values,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', default=','.join(map(str, SEEDS)))
    parser.add_argument('--json-output', default='benchmarks/results/neuron-error-generalization-results.json')
    args = parser.parse_args()
    seeds = [int(x.strip()) for x in args.seeds.split(',') if x.strip()]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError('--seeds must contain unique integers')

    data = load_digits()
    features = data.data.astype(np.float32)
    features /= np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)
    labels = data.target.astype(int)
    guard = load_guard()
    results = [run_seed(features, labels, seed, guard) for seed in seeds]

    metrics = {
        'audit_baseline_accuracy': aggregate([r['known_audit']['frozen_baseline']['accuracy'] for r in results]),
        'audit_after_error_feedback_accuracy': aggregate([r['known_audit']['after_error_feedback']['accuracy'] for r in results]),
        'audit_after_random_control_accuracy': aggregate([r['known_audit']['after_count_class_matched_random_examples']['accuracy'] for r in results]),
        'error_feedback_accuracy_delta_pp': aggregate([r['known_audit']['error_feedback_accuracy_delta_pp'] for r in results]),
        'random_control_accuracy_delta_pp': aggregate([r['known_audit']['random_control_accuracy_delta_pp'] for r in results]),
        'unknown_rejection_baseline': aggregate([r['unknown_audit']['frozen_baseline']['rejection_rate'] for r in results]),
        'unknown_rejection_after_error_feedback': aggregate([r['unknown_audit']['after_error_feedback']['rejection_rate'] for r in results]),
        'unknown_rejection_after_random_control': aggregate([r['unknown_audit']['after_count_class_matched_random_examples']['rejection_rate'] for r in results]),
        'same_failure_pattern_unseen_cases': aggregate([r['same_failure_pattern_generalization']['n_unseen_cases'] for r in results]),
        'same_failure_pattern_fix_rate_error_feedback': aggregate([r['same_failure_pattern_generalization']['error_feedback_fix_rate'] for r in results]),
        'same_failure_pattern_fix_rate_random_control': aggregate([r['same_failure_pattern_generalization']['random_control_fix_rate'] for r in results]),
    }
    report = {
        'status': 'EXECUTED',
        'benchmark': 'Error-feedback generalization test for Neuron metadata guard',
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'dataset': 'sklearn.datasets.load_digits (1,797 real 8x8 grayscale digit images; normalized raw pixel vectors)',
        'guard_source': 'AST-extracted from Untitled36.ipynb',
        'guard_thresholds': {
            'minimum_similarity': float(guard['NEURON_METADATA_MIN_SIMILARITY']),
            'minimum_margin': float(guard['NEURON_METADATA_MIN_MARGIN']),
            'top_exemplars': int(guard['NEURON_METADATA_TOP_EXEMPLARS']),
        },
        'seeds': seeds,
        'protocol': {
            'known_classes': sorted(KNOWN), 'unknown_classes': sorted(UNKNOWN),
            'split': 'Stratified 75/25 train/frozen test; initial memory holds 60% of each known training class; remaining 40% is sequential feedback',
            'update': 'Reveal label after prediction; append correct exemplar only for wrong/rejected samples',
            'control': 'Equal-count, per-class matched random exemplars from non-error feedback samples',
            'generalization': 'Frozen audit examples sharing exact observed true-class, prediction/rejection and rejection-reason signature',
        },
        'aggregate': metrics,
        'per_seed': results,
        'interpretation': 'Small and inconsistent gains are a pilot signal, not proof of robust generalization or human-like learning.',
    }
    output = Path(args.json_output)
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'saved': str(output), 'seeds': seeds, 'aggregate': metrics}, indent=2))


if __name__ == '__main__':
    main()
