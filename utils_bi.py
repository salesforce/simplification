
from utils_categories import get_categories
import numpy as np

edit_categories = get_categories()

def process_bicats_map(bicats_map):
    opis = sorted(bicats_map.keys())
    active_groups = {cat: [] for cat in edit_categories}
    output = []
    for opi in opis:
        for (cat, BI_tag) in bicats_map[opi]:
            if BI_tag == 'B':
                if len(active_groups[cat]) > 0:
                    output.append({"opis": active_groups[cat], "category": cat})
                    active_groups[cat] = []
                active_groups[cat] = [opi]
            else:
                active_groups[cat].append(opi)

    # Add the straggler groups
    for cat, opis in active_groups.items():
        if len(opis) > 0:
            output.append({"opis": opis, "category": cat})
    return output


# Grouping BIO related
def assign_BI_tags(annotations, edits):
    opi2eis = {}
    for idx, anno in enumerate(annotations):
        for opi in anno["edit_idxs"]:
            if opi not in opi2eis:
                opi2eis[opi] = set([])
            opi2eis[opi].add(idx)

    prev_opi = None
    for opi, edit in enumerate(edits):
        edit['idx'] = opi
        if edit["type"] == "equal":
            edit["BI"] = "O"
            continue

        current_eis = opi2eis[opi] # Should all have at least one if completed sample
        previous_eis = opi2eis.get(prev_opi, set([]))
        edit["BI"] = "B" if len(current_eis & previous_eis) == 0 else "I"
        prev_opi = opi


def compute_groups_from_BI(opi2BI):
    assert all(v in ["B", "I"] for v in opi2BI.values())
    groups, current_group = [], []    
    for opi in sorted(opi2BI.keys()):
        BI = opi2BI[opi]
        if BI == "B":
            if len(current_group) > 0:
                groups.append(current_group)
            current_group = [opi]
        else:
            current_group.append(opi)
    if len(current_group) > 0:
        groups.append(current_group)
    return groups

def postprocess_groups(groups, adjacency_only=False):
    categories = sorted(set([g["category"] for g in groups]))
    cat2row = {cat: i for i, cat in enumerate(categories)}
    if len([g for g in groups if len(g["opis"]) > 0]) == 0:
        return []
    max_opi = max([opi for g in groups for opi in g["opis"]])
    A = np.zeros((len(categories), max_opi + 1))
    for g in groups:
        for opi in g["opis"]:
            A[cat2row[g["category"]], opi] = 1
    
    # cats_consec = ['discourse_anaphora_insertion', 'discourse_anaphora_resolution', 'lexical_entity', 'lexical_generic', 'nonsim_fact_correction', 'nonsim_format', 'nonsim_general', 'nonsim_noise_deletion', 'semantic_hypernymy', 'syntactic_generic', 'syntactic_sentence_fusion', 'syntactic_sentence_splitting', 'semantic_deletion']
    cats_nonconsec = ['discourse_reordering', 'nonsim_extraneous_information', 'semantic_elaboration_background', 'semantic_elaboration_example', 'semantic_elaboration_generic', 'syntactic_deletion']

    final_groups = []
    for row, cat in zip(A, categories):
        if cat in cats_nonconsec and not adjacency_only:
            # Merge all of them into a group
            opis = [opi for opi, val in enumerate(row) if val == 1]
            final_groups.append({"opis": opis, "category": cat})
        else:
            # Split into consecutive groups
            current_group = []
            for opi, val in enumerate(row):
                if val == 1:
                    current_group.append(opi)
                else:
                    if len(current_group) > 0:
                        final_groups.append({"opis": current_group, "category": cat})
                        current_group = []
            if len(current_group) > 0:
                final_groups.append({"opis": current_group, "category": cat})
    return final_groups
