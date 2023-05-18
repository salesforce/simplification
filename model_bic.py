from transformers import RobertaForTokenClassification, AutoTokenizer
from utils_bi import process_bicats_map, postprocess_groups
from utils_categories import get_categories
import torch, logging, utils_diff

insert_token_begin = '<insert>'
insert_token_end = '</insert>'
delete_token_begin = '<delete>'
delete_token_end = '</delete>'

edit_categories = get_categories()


class BIC:
    def __init__(self, model_name_or_path, truncation=True, do_postop=False, fill_in_blanks=True, return_groups_only=False, return_scores=False, use_untuned=False, device='cuda'):
        self.tokenizer = AutoTokenizer.from_pretrained("roberta-large")
        self.model = RobertaForTokenClassification.from_pretrained(model_name_or_path)
        self.model.eval().to(device)
        self.device = device

        self.tokenizer.add_special_tokens(
            {'additional_special_tokens': [insert_token_begin, insert_token_end, delete_token_begin, delete_token_end]}
        )
        self.categories = edit_categories

        self.insert_token_begin_id = self.tokenizer.convert_tokens_to_ids(insert_token_begin)
        self.delete_token_begin_id = self.tokenizer.convert_tokens_to_ids(delete_token_begin)

        self.truncation = truncation
        self.do_postop = do_postop
        self.fill_in_blanks = fill_in_blanks
        self.use_untuned = use_untuned

        self.cat_thresholds = {cat: 0.5 for cat in self.categories}
        if not self.use_untuned:
            # if "2898" in model_name_or_path:
            #     self.cat_thresholds = {'lexical_generic': 0.64, 'lexical_entity': 0.23, 'syntactic_generic': 0.76, 'syntactic_sentence_splitting': 0.43, 'syntactic_sentence_fusion': 0.51, 'syntactic_deletion': 0.24, 'discourse_reordering': 0.12, 'discourse_anaphora_resolution': 0.26, 'discourse_anaphora_insertion': 0.37, 'semantic_hypernymy': 0.49, 'semantic_deletion': 0.23, 'semantic_elaboration_generic': 0.17, 'semantic_elaboration_background': 0.18, 'semantic_elaboration_example': 0.09, 'nonsim_format': 0.21, 'nonsim_noise_deletion': 0.13, 'nonsim_fact_correction': 0.26, 'nonsim_extraneous_information': 0.15, 'nonsim_general': 0.37}
            # elif "1449" in model_name_or_path:
            self.cat_thresholds = {'lexical_generic': 0.6, 'lexical_entity': 0.2, 'syntactic_generic': 0.39, 'syntactic_sentence_splitting': 0.27, 'syntactic_sentence_fusion': 0.25, 'syntactic_deletion': 0.27, 'discourse_reordering': 0.11, 'discourse_anaphora_resolution': 0.07, 'discourse_anaphora_insertion': 0.15, 'semantic_hypernymy': 0.07, 'semantic_deletion': 0.63, 'semantic_elaboration_generic': 0.28, 'semantic_elaboration_background': 0.21, 'semantic_elaboration_example': 0.04, 'nonsim_format': 0.4, 'nonsim_noise_deletion': 0.16, 'nonsim_fact_correction': 0.35000000000000003, 'nonsim_extraneous_information': 0.35000000000000003, 'nonsim_general': 0.14}

        self.return_groups_only = return_groups_only
        self.return_scores = return_scores

    def preprocess(self, data):
        batch_texts, batch_op_idxs, batch_opi2type = [], [], []

        for item in data:
            item_edit_texts = []
            op_idxs = []
            opi2type = {}
            for opi, edit in enumerate(item['edits']):
                text = None
                if edit['type'] == 'equal':
                    text = edit['text']
                else:
                    opi2type[opi] = edit['type']
                    op_idxs.append(opi)
                    if edit["type"] == 'insert':
                        text = insert_token_begin + edit['insert'] + insert_token_end
                    elif edit["type"] == 'delete':
                        text = delete_token_begin + edit['delete'] + delete_token_end
                if text is None:
                    raise ValueError(f'Edit type {edit["type"]} not recognized')
                item_edit_texts.append(text)

            batch_texts.append(''.join(item_edit_texts))
            batch_op_idxs.append(op_idxs)
            batch_opi2type.append(opi2type)

        encodings = self.tokenizer(batch_texts, padding=True, truncation=self.truncation)
        token_ids = torch.tensor(encodings['input_ids']).to(self.device)

        batch_opi2toki = []
        for i in range(len(data)):
            tokens = encodings['input_ids'][i]
            op_idxs = batch_op_idxs[i]
            tok_idxs = [i for i in range(len(tokens)) if tokens[i] in [self.insert_token_begin_id, self.delete_token_begin_id]]
            if len(tok_idxs) != len(op_idxs):
                if tokens[-1] == self.tokenizer.pad_token_id:
                    # If this example is padded (i.e. not truncated), the numbers should match up
                    raise ValueError(f'The number of begin tokens does not match the number of edit labels')
                # else:  # Possible truncated
                #     logging.warning('Warning: Mismatch between begin token ids and edits (probably due to truncation) (%d / %d)' % (len(tok_idxs), len(op_idxs)))
            batch_opi2toki.append({opi: toki for opi, toki in zip(op_idxs, tok_idxs)})

        return token_ids, batch_opi2toki, batch_opi2type

    def predict_batch(self, samples):
        token_ids, opi2tokis, opi2types = self.preprocess(samples)
        sigmoid = torch.nn.Sigmoid()

        with torch.no_grad():
            outputs = self.model(token_ids)
            logits = outputs.logits
            logits_cats = logits[:, :, :len(self.categories)]
            logits_is_I = logits[:, :, len(self.categories):2*len(self.categories)]
            logits_is_B = logits[:, :, 2*len(self.categories):]
            probs_cats = sigmoid(logits_cats)
            probs_is_I = sigmoid(logits_is_I)
            probs_is_B = sigmoid(logits_is_B)

        batch_output, batch_cat_scores, batch_bi_scores = [], [], []
        for opi2toki, prob_cats, prob_is_I, prob_is_B, opi2type in zip(opi2tokis, probs_cats, probs_is_I, probs_is_B, opi2types):
            opi2bicats = {}
            cat_scores, bi_scores = {}, {}
            for opi, toki in opi2toki.items():
                opi2bicats[opi] = []
                max_p_cat = prob_cats[toki].max()
                cat_scores[opi] = prob_cats[toki].tolist()
                bi_scores[opi] = prob_is_I[toki].tolist()
                for cat, p_cat, p_is_I, p_is_B in zip(self.categories, prob_cats[toki], prob_is_I[toki], prob_is_B[toki]):
                    thresh = self.cat_thresholds[cat]
                    if p_cat > thresh:
                        BI_tag = 'I' if p_is_I > p_is_B else 'B'
                        opi2bicats[opi].append((cat, BI_tag))
            batch_cat_scores.append(cat_scores)
            batch_bi_scores.append(bi_scores)
            output = process_bicats_map(opi2bicats)
            if self.fill_in_blanks:
                missing_opis = set(opi2type.keys()) - set(opi2toki.keys())
                if len(missing_opis) > 0:
                    for opi in missing_opis:
                        opi_type = opi2type[opi]
                        fill_in_cat = "semantic_deletion" if opi_type == "delete" else "nonsim_extraneus_information"
                        output.append({"opis": [opi], "category": fill_in_cat})

            if self.do_postop:
                output = postprocess_groups(output)
            if self.return_groups_only:
                output = [g["opis"] for g in output]

            batch_output.append(output)

        for sample, output in zip(samples, batch_output):
            for group in output:
                edits = sample["edits"]
                if "semantic_elaboration" in group["category"] and all([edits[opi]["type"] == "delete" for opi in group["opis"]]):
                    group["category"] = "semantic_deletion"

        if self.return_scores:
            return batch_cat_scores, batch_bi_scores, batch_output

        # Resort by minimum opi
        batch_output = [sorted(g, key=lambda x: min(x["opis"])-0.01*max(x["opis"])) for g in batch_output]

        return batch_output

    def predict(self, samples, batch_size=32):
        output = []
        for i in range(0, len(samples), batch_size):
            batch_out = self.predict_batch(samples[i:i+batch_size])
            output += batch_out
        return output

    def predict_one(self, sample):
        output = self.predict([sample])
        if self.return_scores:
            return output[0][0], output[1][0], output[2][0]
        return output[0]

    def predict_from_text_pair(self, text1, text2):
        edits = utils_diff.get_edit_operations(text1, text2, split_replace=True, split_sentences=True)
        return self.predict_one({"edits": edits})


if __name__ == "__main__":
    from utils_eval import compute_type_score
    import json

    with open("data/simple_wiki_edits_0.5_val.json", "r") as f:
        dataset_val = json.load(f)

    model = BIC("/export/share/jvig/simplification-private/output/eot-bi3-v5/checkpoint-1932/")

    D = dataset_val[0]
    pred = model.predict([D])
    print(compute_type_score([D], pred, high_level=False))
    print(compute_type_score([D], pred, high_level=True))
