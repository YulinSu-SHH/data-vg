import os
import os.path as osp
import json
import pdb
import random
from collections import defaultdict
from sys import prefix
from petrel_client.client import Client
from tqdm import tqdm
import csv

conf_path = '~/petreloss.conf'
client = Client(conf_path)

anno_path = '/mnt/lustre/suyulin/ofa-hf/suyulin_data/genome/region_descriptions.json'
ceph_root = 'zz1424:s3://publicdataset_8/Visual_Genome_Dataset_V1.2/unzip/data/'
img_code_root='/mnt/lustre/suyulin/ofa-hf/suyulin_data/genome/pure_img_code/'

out_tsv_label = '/mnt/lustre/suyulin/ofa-hf/suyulin_data/genome/genome_label.tsv'
out_tsv_unlabel = '/mnt/lustre/suyulin/ofa-hf/suyulin_data/genome/genome_unlabel.tsv'

with open(anno_path,'r') as f_in:
    vg_list = json.load(f_in)

def xywh2xyxy(bbox): 
    x1, y1, w, h = bbox
    x2 = x1 + w
    y2 = y1 + h
    return [x1, y1, x2, y2]

def clean_ref(ref):
    ref = ref.replace('\r', '')
    ref = ref.replace('\n', '')
    ref = ref.replace('\t', ' ')
    return ref

sub_set1 = 'zz1424:s3://publicdataset_8/Visual_Genome_Dataset_V1.2/unzip/data/images/VG_100K'
sub_set2 = 'zz1424:s3://publicdataset_8/Visual_Genome_Dataset_V1.2/unzip/data/images2/VG_100K_2'

##### item 1
uniq_id = 0

##### item 3
caption = ''

##### item 4
question = ''

##### item 6
gt_objects = ''

##### item 7
dataset_name = 'genome'

##### item 8
task = 'vg'  # Generalized Visual Grounding (gvg)

separator = '\t'

f_out_label= open(out_tsv_label, 'w', newline='')
f_out_unlabel= open(out_tsv_unlabel, 'w', newline='')
tsv_writer_label = csv.writer(f_out_label, delimiter=separator)
tsv_writer_unlabel = csv.writer(f_out_unlabel, delimiter=separator)
for line_idx, line in enumerate(vg_list):
    filename = str(line['id']) + '.jpg'
    file_rale_path = osp.join('images/VG_100K', filename)
    ceph_path_candidate = osp.join(ceph_root, file_rale_path)
    ##### item 2
    if client.contains(ceph_path_candidate):
        ceph_path=ceph_path_candidate
        prefix="data1_"
    else:
        file_rale_path = osp.join('images2/VG_100K_2', filename)
        ceph_path = osp.join(ceph_root, file_rale_path)
        prefix="data2_"
        assert client.contains(ceph_path) , f"Please check image at {filename}"
    # ceph_path = osp.join(ceph_root, file_rale_path)
    txtfilename=prefix+filename[:-4]+'.txt'
    ##### item 3
    img_code_path=osp.join(img_code_root, txtfilename)

    if client.contains(ceph_path):
        for iregion in line['regions']:
            ##### item 5
            refs = iregion['phrase']
            refs = clean_ref(refs)

            bbox = [iregion['x'], iregion['y'], iregion['width'], iregion['height']]
            gt_objects = ','.join([str(x) for x in xywh2xyxy(bbox)])

            res_line = [uniq_id, ceph_path, img_code_path, caption, refs, gt_objects, dataset_name, task]
            if line_idx<len(vg_list)/3:
                tsv_writer_label.writerow(res_line)
            else:
                tsv_writer_unlabel.writerow(res_line)
            uniq_id += 1    

    if uniq_id % 10000 == 0:
        print(f"{line_idx} of {len(vg_list)}", flush=True)
