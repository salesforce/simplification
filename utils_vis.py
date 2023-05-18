import utils_diff

def visualize_edit_group(edits, edit_group):
    m, M = min(edit_group["opis"]), max(edit_group["opis"])
    opi_range = list(range(m, M+1))
    if opi_range[0] > 0 and edits[opi_range[0]-1]["type"] == "equal":
        opi_range = [opi_range[0]-1] + opi_range
    if opi_range[-1] < len(edits)-1 and edits[opi_range[-1]+1]["type"] == "equal":
        opi_range = opi_range + [opi_range[-1]+1]

    sub_edits = [edits[opi] for opi in opi_range]
    diff_text = utils_diff.make_colored_text(from_ops=sub_edits)
    if opi_range[0] > 0:
        diff_text = "[...] " + diff_text
    if opi_range[-1] < len(edits)-1:
        diff_text = diff_text + " [...]"
    return "[%s] %s" % (edit_group["category"].ljust(30), diff_text)

def visualize_edit_groups(input_text, output_text, edit_groups):
    print("There are a total of %d identified groups." % (len(edit_groups)))
    edits = utils_diff.get_edit_operations(input_text, output_text, split_replace=True, split_sentences=True)
    for ed_group in edit_groups:
        print(visualize_edit_group(edits, ed_group))