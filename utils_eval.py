from sklearn.metrics import precision_recall_fscore_support
from collections import Counter, defaultdict
import itertools, numpy as np

def compute_group_score(annotations, pred_groups):
    # Similar to adjusted rand index
    # pred_groups format: [(1,2,3), (5,6), (7,8)]

    ref_groups = {}
    for gidx, anno in enumerate(annotations):
        for idx in anno["edit_idxs"]:
            if idx not in ref_groups:
                ref_groups[idx] = []
            ref_groups[idx].append(gidx)

    pred_groups_map = {}
    for idx, vs in enumerate(pred_groups):
        for v in vs:
            if v not in pred_groups_map:
                pred_groups_map[v] = []
            pred_groups_map[v].append(idx)
    num_pred_groups = len(pred_groups)

    # Fill in gaps into individual groups
    for v in ref_groups.keys():
        if v not in pred_groups_map:
            pred_groups_map[v] = [num_pred_groups]
            num_pred_groups += 1    

    assert set(ref_groups.keys()) == set(pred_groups_map.keys())

    agreement = []
    edit_idxs = ref_groups.keys()
    for idx1, idx2 in itertools.combinations(edit_idxs, 2):
        is_any_common_group1 = len(set(ref_groups[idx1]) & set(ref_groups[idx2])) > 0
        is_any_common_group2 = len(set(pred_groups_map[idx1]) & set(pred_groups_map[idx2])) > 0
        if is_any_common_group1 and is_any_common_group2:
            agreement.append(1)
        elif not is_any_common_group1 and not is_any_common_group2:
            agreement.append(1)
        else:
            agreement.append(0)
    return np.mean(agreement)


def compute_type_score(dataset, predictions, high_level=False):
    # dataset format: the validation dataset (with each element containing an "annotations" key)
    # predictions format:
    # [{1: ["syntactic_sentence_splitting", "semantic_deletion"], 2: ["semantic_deletion"], 3: ["semantic_deletion"]}, ...]
    assert len(dataset) == len(predictions), "The number of predictions should be the same as the number of examples (%d vs. %d)" % (len(predictions), len(dataset))

    category_names = ['discourse_anaphora_insertion', 'discourse_anaphora_resolution', 'discourse_reordering', 'lexical_entity', 'lexical_generic', 'nonsim_extraneous_information', 'nonsim_fact_correction', 'nonsim_format', 'nonsim_general', 'nonsim_noise_deletion', 'semantic_deletion', 'semantic_elaboration_background', 'semantic_elaboration_example', 'semantic_elaboration_generic', 'semantic_hypernymy', 'syntactic_deletion', 'syntactic_generic', 'syntactic_sentence_fusion', 'syntactic_sentence_splitting']
    class_names = ["discourse", "lexical", "nonsim", "semantic", "syntactic"]
    label_names = category_names if not high_level else class_names

    labels, preds = [], []
    for d, pred_groups in zip(dataset, predictions):
        pred = {}
        for pred_group in pred_groups:
            for opi in pred_group["opis"]:
                if opi not in pred:
                    pred[opi] = []
                pred[opi].append(pred_group["category"])

        ref_types = {}
        for anno in d["annotations"]:
            for opi in anno["edit_idxs"]:
                if opi not in ref_types:
                    ref_types[opi] = []
                ref_types[opi].append(anno["edit_type"])

        if high_level:
            ref_types = {k: [v.split("_")[0] for v in vs] for k, vs in ref_types.items()}
            pred = {k: [v.split("_")[0] for v in vs] for k, vs in pred.items()}

        for k in ref_types.keys():
            labels.append([0 if v not in ref_types[k] else 1 for v in label_names])
            preds.append([0 if v not in pred.get(k, []) else 1 for v in label_names])

    precision, recall, f1, _ = precision_recall_fscore_support(np.array(labels), np.array(preds), average='weighted')
    return {"precision": precision, "recall": recall, "f1": f1}


def compute_type_score_by_class(dataset, predictions, high_level=False):
    # dataset format: the validation dataset (with each element containing an "annotations" key)
    # predictions format:
    # [{1: ["syntactic_sentence_splitting", "semantic_deletion"], 2: ["semantic_deletion"], 3: ["semantic_deletion"]}, ...]
    assert len(dataset) == len(predictions), "The number of predictions should be the same as the number of examples"

    category_names = ['discourse_anaphora_insertion', 'discourse_anaphora_resolution', 'discourse_reordering', 'lexical_entity', 'lexical_generic', 'nonsim_extraneous_information', 'nonsim_fact_correction', 'nonsim_format', 'nonsim_general', 'nonsim_noise_deletion', 'semantic_deletion', 'semantic_elaboration_background', 'semantic_elaboration_example', 'semantic_elaboration_generic', 'semantic_hypernymy', 'syntactic_deletion', 'syntactic_generic', 'syntactic_sentence_fusion', 'syntactic_sentence_splitting']
    class_names = ["discourse", "lexical", "nonsim", "semantic", "syntactic"]
    label_names = category_names if not high_level else class_names

    class_to_labels = defaultdict(list)
    class_to_preds = defaultdict(list)
    for d, pred_groups in zip(dataset, predictions):
        pred = {}
        for pred_group in pred_groups:
            for opi in pred_group["opis"]:
                if opi not in pred:
                    pred[opi] = []
                pred[opi] += pred_group["category"]

        ref_types = {}
        for anno in d["annotations"]:
            for idx in anno["edit_idxs"]:
                if idx not in ref_types:
                    ref_types[idx] = []
                ref_types[idx].append(anno["edit_type"])

        for k in ref_types.keys():
            for v in label_names:
                class_to_labels[v].append(0 if v not in ref_types[k] else 1)
                class_to_preds[v].append(0 if v not in pred.get(k, []) else 1)

    results = []
    for cls in label_names:
        labels = class_to_labels[cls]
        preds = class_to_preds[cls]
        precision, recall, f1, support = precision_recall_fscore_support(np.array(labels), np.array(preds), average='binary')
        results.append(
            {"class": cls, "precision": precision, "recall": recall, "f1": f1, 'count_true': int(sum(labels)), 'count_pred': int(sum(preds))}
        )
    return results


def compute_exact_score(dataset, predictions, partial=False):
    # Groups format: {(1,2,3): "syntactic_sentence_splitting", (3,4,5): "semantic_deletion"}
    all_scores = []
    for d, pred_groups in zip(dataset, predictions):
        ref_groups = {tuple(anno["edit_idxs"]): anno["edit_type"] for anno in d["annotations"]}
        ref_groups = {tuple(sorted(k)): v for k, v in ref_groups.items()}
        pred_groups = {tuple(sorted(pred_group["opis"])): pred_group["category"] for pred_group in pred_groups}

        agreements = []
        for ref_group_ids, ref_cat in ref_groups.items():
            score = 0.0
            for pred_group, pred_cat in pred_groups.items():
                if ref_cat != pred_cat:
                    continue
                if partial:
                    jaccard = len(set(ref_group_ids).intersection(set(pred_group))) / len(set(ref_group_ids).union(set(pred_group)))
                    if jaccard > 0.5:
                        score = 1.0
                else:
                    if set(ref_group_ids) == set(pred_group):
                        score = 1.0
            agreements.append(score)
        all_scores.append(np.mean(agreements))
    return np.mean(all_scores)
