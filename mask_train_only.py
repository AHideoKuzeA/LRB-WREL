import json
import pickle
import random
import copy
import os
from collections import defaultdict

random.seed(42)


input_json_path = "/data/rrsisd/instances.json"
input_refs_path = "/data/rrsisd/refs(unc).p"
output_dir = "/data/rrsisd-mask"
os.makedirs(output_dir, exist_ok=True)


mask_ratios = [0.1, 0.3, 0.5, 0.7, 0.9]


with open(input_json_path, "r") as f:
    original_data = json.load(f)
with open(input_refs_path, "rb") as f:
    ref_data = pickle.load(f)



categories = {cat["id"]: cat["name"] for cat in original_data["categories"]}


ann_by_cat = defaultdict(list)
for ann in original_data["annotations"]:
    ann_by_cat[ann["categories_id"]].append(ann)


train_refs = [ref for ref in ref_data if ref["split"] == "train"]
val_refs = [ref for ref in ref_data if ref["split"] == "val"]
test_refs = [ref for ref in ref_data if ref["split"] == "test"]


train_annid_to_ref = {ref["ann_id"]: ref for ref in train_refs}
train_ann_ids = set(train_annid_to_ref.keys())


train_anns = [ann for ann in original_data["annotations"] if ann["id"] in train_ann_ids]
val_anns = [ann for ann in original_data["annotations"] if ann["id"] not in train_ann_ids and ann.get("image_id") in {ref["image_id"] for ref in val_refs}]
test_anns = [ann for ann in original_data["annotations"] if ann["id"] not in train_ann_ids and ann.get("image_id") in {ref["image_id"] for ref in test_refs}]



for ratio in mask_ratios:
    train_anns_copy = copy.deepcopy(train_anns)
    train_refs_copy = copy.deepcopy(train_refs)


    total_mask = int(len(train_anns_copy) * ratio)
    remaining_mask = total_mask
    masked_anns = []


    ann_by_cat_in_train = defaultdict(list)
    for ann in train_anns_copy:
        ann_by_cat_in_train[ann["categories_id"]].append(ann)

    for cat_id, anns in sorted(ann_by_cat_in_train.items(), key=lambda x: len(x[1]), reverse=True):
        if remaining_mask <= 0:
            break
        n = max(1, int(ratio * len(anns)))
        n = min(n, remaining_mask, len(anns))
        masked_anns.extend(random.sample(anns, n))
        remaining_mask -= n

    masked_ids = set(ann["id"] for ann in masked_anns)
    id_to_cat_id = {ann["id"]: ann["categories_id"] for ann in masked_anns}
    annid_to_original_expr = {}


    train_anns_nomask = [ann for ann in train_anns_copy if ann["id"] not in masked_ids]
    train_anns_mask = [ann for ann in train_anns_copy if ann["id"] in masked_ids]

    train_refs_nomask = []
    train_refs_mask = []

    for ref in train_refs_copy:
        ann_id = ref["ann_id"]
        if ann_id in masked_ids:
            cat_name = categories[id_to_cat_id[ann_id]]
            ref["sentences"][0]["raw"] = cat_name
            ref["sentences"][0]["tokens"] = cat_name.split()
            train_refs_mask.append(ref)
        else:
            train_refs_nomask.append(ref)


    all_anns = train_anns_nomask + train_anns_mask + val_anns + test_anns
    all_refs = train_refs_nomask + train_refs_mask + val_refs + test_refs

    ratio_str = str(int(ratio * 100))
    def save_json(data, name): 
        with open(os.path.join(output_dir, name), "w") as f:
            json.dump({
                **original_data,
                "annotations": data,
            }, f, indent=2)

    def save_pkl(data, name):
        with open(os.path.join(output_dir, name), "wb") as f:
            pickle.dump(data, f)


    save_json(train_anns_nomask + val_anns + test_anns, f"instances-{ratio_str}-no-mask.json")
    save_pkl(train_refs_nomask + val_refs + test_refs, f"refs(unc)-{ratio_str}-no-mask.p")


    save_json(train_anns_mask + val_anns + test_anns, f"instances-{ratio_str}-mask.json")
    save_pkl(train_refs_mask + val_refs + test_refs, f"refs(unc)-{ratio_str}-mask.p")


    save_json(all_anns, f"instances-{ratio_str}-all.json")
    save_pkl(all_refs, f"refs(unc)-{ratio_str}-all.p")

    sample_to_show = random.sample(train_anns_mask, min(5, len(train_anns_mask)))
    for i, ann in enumerate(sample_to_show, 1):
        ann_id = ann["id"]
        image_id = ann["image_id"]
        bbox = ann["bbox"]
        cat_id = ann["categories_id"]
        cat_name = categories[cat_id]
        ref = train_annid_to_ref[ann_id]
        ref_id = ref["ref_id"]


        original_sent = copy.deepcopy(ref["sentences"][0])
        original_expr = original_sent["raw"]
        original_tokens = original_sent["tokens"]


        replaced_expr = cat_name
        replaced_tokens = cat_name.split()

