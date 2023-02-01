import argparse
import json
from operator import imod
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple
from xml.etree.ElementTree import parse
import os.path as osp
import csv

import numpy as np
import torch
import xmltodict
from torchvision.ops.boxes import box_area
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser("Conversion script")

    parser.add_argument(
        "--flickr_path",
        default="/mnt/lustre/share_data/suyulin/VG/flickr30k/flickr30k_annos",
        type=str,
        help="Path to the flickr dataset",
    )
    parser.add_argument(
        "--out_path",
        default="/mnt/lustre/share_data/suyulin/VG/flickr30k",
        type=str,
        help="Path where to export the resulting dataset.",
    )

    parser.add_argument(
        "--ceph_root",
        default="zz1424:s3://production-public-flickr_image/Flickr_Image/unzip/flickr30k_images/flickr30k_images/",
        type=str,
        help="ceph_root for the img data",
    )

    parser.add_argument(
        "--merge_ground_truth",
        action="store_true",
        help="Whether to follow Bryan Plummer protocol and merge ground truth. By default, all the boxes for an entity are kept separate",
    )

    return parser.parse_args()


def box_xywh_to_xyxy(x):
    """Accepts a list of bounding boxes in coco format (xmin,ymin, width, height)
    Returns the list of boxes in pascal format (xmin,ymin,xmax,ymax)

    The boxes are expected as a numpy array
    """
    result = x.copy()
    result[..., 2:] += result[..., :2]
    return result


def xyxy2xywh(box: List):
    """Accepts a list of bounding boxes in pascal format (xmin,ymin,xmax,ymax)
    Returns the list of boxes in coco format (xmin,ymin, width, height)
    """
    xmin, ymin, xmax, ymax = box
    h = ymax - ymin
    w = xmax - xmin
    return [xmin, ymin, w, h]


#### The following loading utilities are imported from
#### https://github.com/BryanPlummer/flickr30k_entities/blob/68b3d6f12d1d710f96233f6bd2b6de799d6f4e5b/flickr30k_entities_utils.py
# Changelog:
#    - Added typing information
#    - Completed docstrings


def get_sentence_data(filename) -> List[Dict[str, Any]]:
    """
    Parses a sentence file from the Flickr30K Entities dataset

    input:
      filename - full file path to the sentence file to parse

    output:
      a list of dictionaries for each sentence with the following fields:
          sentence - the original sentence
          phrases - a list of dictionaries for each phrase with the
                    following fields:
                      phrase - the text of the annotated phrase
                      first_word_index - the position of the first word of
                                         the phrase in the sentence
                      phrase_id - an identifier for this phrase
                      phrase_type - a list of the coarse categories this
                                    phrase belongs to

    """
    with open(filename, "r") as f:
        sentences = f.read().split("\n")

    annotations = []
    for sentence in sentences:
        if not sentence:
            continue

        first_word = []
        phrases = []
        phrase_id = []
        phrase_type = []
        words = []
        current_phrase = []
        add_to_phrase = False
        for token in sentence.split():
            if add_to_phrase:
                if token[-1] == "]":
                    add_to_phrase = False
                    token = token[:-1]
                    current_phrase.append(token)
                    phrases.append(" ".join(current_phrase))
                    current_phrase = []
                else:
                    current_phrase.append(token)

                words.append(token)
            else:
                if token[0] == "[":
                    add_to_phrase = True
                    first_word.append(len(words))
                    parts = token.split("/")
                    phrase_id.append(parts[1][3:])
                    phrase_type.append(parts[2:])
                else:
                    words.append(token)

        sentence_data = {"sentence": " ".join(words), "phrases": []}
        for index, phrase, p_id, p_type in zip(first_word, phrases, phrase_id, phrase_type):
            sentence_data["phrases"].append(
                {"first_word_index": index, "phrase": phrase, "phrase_id": p_id, "phrase_type": p_type}
            )

        annotations.append(sentence_data)

    return annotations


def convert(
    subset: str, flickr_path: Path, output_path: Path, merge_ground_truth: bool, next_img_id: int = 0, next_id: int = 0, ceph_root: str = 'zz1424:s3://production-public-flickr_image/Flickr_Image/unzip/flickr30k_images/flickr30k_images/'
):

    with open(flickr_path / f"{subset}.txt") as fd:
        ids = [int(l.strip()) for l in fd]

    multibox_entity_count = 0

    categories = [{"supercategory": "object", "id": 1, "name": "object"}]
    annotations = []
    images = []

    print(f"Exporting {subset}...")
    global_phrase_id = 0
    global_phrase_id2phrase = {}

    with open(osp.join(output_path, f"{subset}.tsv"), 'w', newline='') as f_out:
        tsv_writer = csv.writer(f_out, delimiter='\t')
        for img_id in tqdm(ids):
            with open(flickr_path / "Annotations" / f"{img_id}.xml") as xml_file:
                annotation = xmltodict.parse(xml_file.read())["annotation"]

            anno_file = os.path.join(flickr_path, "Annotations/%d.xml" % img_id)

            # Parse Annotation
            root = parse(anno_file).getroot()
            obj_elems = root.findall("./object")
            target_bboxes = {}

            for elem in obj_elems:
                if elem.find("bndbox") == None or len(elem.find("bndbox")) == 0:
                    continue
                xmin = float(elem.findtext("./bndbox/xmin"))
                ymin = float(elem.findtext("./bndbox/ymin"))
                xmax = float(elem.findtext("./bndbox/xmax"))
                ymax = float(elem.findtext("./bndbox/ymax"))
                assert 0 < xmin and 0 < ymin

                xyxy_box = [xmin, ymin, xmax, ymax]

                for name in elem.findall("name"):
                    entity_id = int(name.text)
                    assert 0 < entity_id
                    if not entity_id in target_bboxes:
                        target_bboxes[entity_id] = []
                    else:
                        multibox_entity_count += 1
                    # Dict from entity_id to list of all the bounding boxes
                    target_bboxes[entity_id].append(xyxy_box)

            if merge_ground_truth:
                merged_bboxes = defaultdict(list)
                for eid, bbox_list in target_bboxes.items():
                    boxes_xyxy = torch.as_tensor(bbox_list, dtype=torch.float)
                    gt_box_merged = [
                        min(boxes_xyxy[:, 0]).item(),
                        min(boxes_xyxy[:, 1]).item(),
                        max(boxes_xyxy[:, 2]).item(),
                        max(boxes_xyxy[:, 3]).item(),
                    ]
                    merged_bboxes[eid] = gt_box_merged

                target_bboxes = merged_bboxes

            sents = get_sentence_data(flickr_path / "Sentences" / f"{img_id}.txt")
            for sent_id, sent in enumerate(sents):

                spans = {}  # global phrase ID to span in sentence
                phraseid2entityid = {}
                entityid2phraseid = defaultdict(list)
                sentence = sent["sentence"]
                entity_ids = [int(p["phrase_id"]) for p in sent["phrases"]]

                for global_phrase_id, phrase in enumerate(sent["phrases"]):
                    phraseid2entityid[global_phrase_id] = int(phrase["phrase_id"])
                    entityid2phraseid[int(phrase["phrase_id"])].append(global_phrase_id)
                    first_word = phrase["first_word_index"]
                    beg = sum([len(x) for x in sentence.split()[:first_word]]) + first_word
                    spans[global_phrase_id] = (beg, beg + len(phrase["phrase"]))
                    assert sentence[beg : beg + len(phrase["phrase"])] == phrase["phrase"]

                tokens_positive_eval = []
                tokens_positive_bbox = []
                for gpid, span in spans.items():
                    if phraseid2entityid[gpid] in target_bboxes:
                        tokens_positive_eval.append(span)
                        tokens_positive_bbox.append(target_bboxes[phraseid2entityid[gpid]])

                spans_str_list = [','.join([str(x) for x in tu]) for tu in tokens_positive_eval]
                spans_str = ';'.join(spans_str_list)

                bbox_str_list = [','.join([','.join([str(c) for c in x]) for x in tu]) for tu in tokens_positive_bbox]
                bbox_str = ';'.join(bbox_str_list)
                
                # item 1
                uniq_id = img_id
                # item 2
                ceph_path = osp.join(ceph_root, f"{img_id}.jpg")
                # item 3
                caption = sentence + '<&noun&>' + spans_str
                # item 4
                question = ''
                # item 5
                refs = bbox_str
                # item 6
                gt_objects = ''
                # item 7
                dataset_name = 'flickr30k'
                # item 8
                task = 'vg'

                res_line = [uniq_id, ceph_path, caption, question, refs, gt_objects, dataset_name, task]
                tsv_writer.writerow(res_line)


    return next_img_id, next_id


def main(args):
    flickr_path = Path(args.flickr_path)
    output_path = Path(args.out_path)
    os.makedirs(str(output_path), exist_ok=True)

    next_img_id, next_id = convert("train", flickr_path, output_path, args.merge_ground_truth)
    next_img_id, next_id = convert("val", flickr_path, output_path, args.merge_ground_truth, next_img_id, next_id)
    next_img_id, next_id = convert("test", flickr_path, output_path, args.merge_ground_truth, next_img_id, next_id)

if __name__ == "__main__":
    main(parse_args())
