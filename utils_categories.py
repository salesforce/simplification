categorization = {
    "Lexical": ["Generic", "Entity"],
    "Syntactic": ["Generic", "Sentence Splitting", "Sentence Fusion", "Deletion"],
    "Discourse": ["Reordering", "Anaphora Resolution", "Anaphora Insertion"],
    "Semantic": ["Elaboration - Generic", "Elaboration - Background", "Elaboration - Example", "Deletion", "Hypernymy"],
    "NonSim": ["Format", "Noise Deletion", "Fact Correction", "Extraneous Information", "General"],
}
consolidated_categorization = {
    "Lexical": ["Lexical Edits"],
    "Syntactic": ["Syntax Edit", "Sent Split", "Sent Fusion"],
    "Discourse": ["Reordering", "Anaphoras"],
    "Semantic": ["Deletions", "Elaborations"],
    "NonSim": ["Extra Info", "Format"]
}


def get_full_label(group, edit_type):
    group = group.lower()
    return "%s_%s" % (group, edit_type.lower().replace(" - ", "_").replace(" ", "_"))


def edit2consolidated(category):
    category = category.lower()
    if "nonsim_extraneous_information" == category:
        return "Extra Info"
    if "elaboration" in category:
        return "Elaborations"
    if category in ["semantic_deletion", "syntactic_deletion", "noise_deletion"]:
        return "Deletions"
    if "anaphora" in category:
        return "Anaphoras"
    if "lexical" in category:
        return "Lexical Edits"
    if "splitting" in category:
        return "Sent Split"
    if "fusion" in category:
        return "Sent Fusion"
    if "format" in category:
        return "Format"
    if "reordering" in category:
        return "Reordering"
    if category == "syntactic_generic":
        return "Syntax Edit"
    return category


def consolidate_annotations(annotations):
    new_annotations = []
    for anno in annotations:
        if anno["category"] in ["semantic_hypernymy", "nonsim_general", "nonsim_fact_correction", "nonsim_noise_deletion"]:
            continue
        new_anno = {"opis": anno["opis"], "category": edit2consolidated(anno["category"])}
        new_annotations.append(new_anno)
    # print(new_annotations)
    return new_annotations


def get_figure_name(group, edit_type):
    group = group.lower()
    label_name = edit_type
    if group == "lexical" and label_name == "Generic":
        label_name = "Phrasal Edit"
    elif group == "syntactic" and label_name == "Generic":
        label_name = "Syntactic Edit"
    elif group == "lexical" and label_name == "Entity":
        label_name = "Entity Edit"

    label_name = label_name.replace("Elaboration", "Elab.").replace("Background", "Bkgd.").replace("Example", "Ex.").replace("Generic", "Gen.").replace(" - ", " ")
    label_name = label_name.replace("Extraneous", "Extra").replace("Information", "Info.")
    label_name = label_name.replace("Resolution", "Res.").replace("Insertion", "Ins.")
    label_name = label_name.replace("Sentence", "Sent.").replace("Splitting", "Split").replace("Fusion", "Fus.")
    label_name = label_name.replace("Correction", "Correct")#.replace("Deletion", "Del.")
    label_name = label_name.replace("Noise Deletion", "Noise Del.")
    label_name = label_name.replace("Anaphora", "Anaph.").replace("Syntactic", "Synt.")
    return label_name


def get_categories():
    flat_categories = []
    for group, labels in categorization.items():
        for label in labels:
            flat_categories.append(
                "%s_%s" % (group.lower(), label.lower().replace(" - ", "_").replace(" ", "_"))
            )
    return flat_categories
