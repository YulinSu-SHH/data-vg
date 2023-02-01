import os
import os.path as osp
import pdb
import json
import csv
from collections import defaultdict
from tqdm import tqdm
import random
import cv2
import numpy as np
from os.path import splitext
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from random import choice, sample
from collections import defaultdict
import h5py
from petrel_client.client import Client
conf_path = '~/petreloss.conf'
client = Client(conf_path)


def read_img_ceph(img_path):
    # img_path= os.path.join(img_path)
    img_bytes = client.get(img_path)
    assert(img_bytes is not None)
    img_mem_view = memoryview(img_bytes)
    img_array= np.frombuffer(img_mem_view, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    return img

def colormap(rgb=False):
    color_list = np.array([
        0.000, 0.447, 0.741, 0.850, 0.325, 0.098, 0.929, 0.694, 0.125, 0.494, 0.184, 0.556, 0.466, 0.674, 0.188, 0.301,
        0.745, 0.933, 0.635, 0.078, 0.184, 0.300, 0.300, 0.300, 0.600, 0.600, 0.600, 1.000, 0.000, 0.000, 1.000, 0.500,
        0.000, 0.749, 0.749, 0.000, 0.000, 1.000, 0.000, 0.000, 0.000, 1.000, 0.667, 0.000, 1.000, 0.333, 0.333, 0.000,
        0.333, 0.667, 0.000, 0.333, 1.000, 0.000, 0.667, 0.333, 0.000, 0.667, 0.667, 0.000, 0.667, 1.000, 0.000, 1.000,
        0.333, 0.000, 1.000, 0.667, 0.000, 1.000, 1.000, 0.000, 0.000, 0.333, 0.500, 0.000, 0.667, 0.500, 0.000, 1.000,
        0.500, 0.333, 0.000, 0.500, 0.333, 0.333, 0.500, 0.333, 0.667, 0.500, 0.333, 1.000, 0.500, 0.667, 0.000, 0.500,
        0.667, 0.333, 0.500, 0.667, 0.667, 0.500, 0.667, 1.000, 0.500, 1.000, 0.000, 0.500, 1.000, 0.333, 0.500, 1.000,
        0.667, 0.500, 1.000, 1.000, 0.500, 0.000, 0.333, 1.000, 0.000, 0.667, 1.000, 0.000, 1.000, 1.000, 0.333, 0.000,
        1.000, 0.333, 0.333, 1.000, 0.333, 0.667, 1.000, 0.333, 1.000, 1.000, 0.667, 0.000, 1.000, 0.667, 0.333, 1.000,
        0.667, 0.667, 1.000, 0.667, 1.000, 1.000, 1.000, 0.000, 1.000, 1.000, 0.333, 1.000, 1.000, 0.667, 1.000, 0.167,
        0.000, 0.000, 0.333, 0.000, 0.000, 0.500, 0.000, 0.000, 0.667, 0.000, 0.000, 0.833, 0.000, 0.000, 1.000, 0.000,
        0.000, 0.000, 0.167, 0.000, 0.000, 0.333, 0.000, 0.000, 0.500, 0.000, 0.000, 0.667, 0.000, 0.000, 0.833, 0.000,
        0.000, 1.000, 0.000, 0.000, 0.000, 0.167, 0.000, 0.000, 0.333, 0.000, 0.000, 0.500, 0.000, 0.000, 0.667, 0.000,
        0.000, 0.833, 0.000, 0.000, 1.000, 0.000, 0.000, 0.000, 0.143, 0.143, 0.143, 0.286, 0.286, 0.286, 0.429, 0.429,
        0.429, 0.571, 0.571, 0.571, 0.714, 0.714, 0.714, 0.857, 0.857, 0.857, 1.000, 1.000, 1.000
    ]).astype(np.float32)
    color_list = color_list.reshape((-1, 3)) * 255
    if not rgb:
        color_list = color_list[:, ::-1]
    return color_list

def pre_caption(caption, max_words):
    caption = caption.lower().lstrip(",.!?*#:;~").replace('-', ' ').replace('/', ' ').replace('<person>', 'person')

    caption = re.sub(
        r"\s{2,}",
        ' ',
        caption,
    )
    caption = caption.rstrip('\n')
    caption = caption.strip(' ')

    # truncate caption
    caption_words = caption.split(' ')
    if len(caption_words) > max_words:
        caption = ' '.join(caption_words[:max_words])

    return caption

def vis_bbox_with_text(box, label, text, ax, color_list, box_alpha=1):
    x1, y1, x2, y2 = box
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    color_box = color_list[label % len(color_list), 0:3]
    ax.add_patch(
        plt.Rectangle((box[0], box[1]),
                      box[2] - box[0],
                      box[3] - box[1],
                      fill=True,
                      color=color_box,
                      linewidth=0.5,
                      alpha=box_alpha * 0.2))
    box_w = x2 - x1
    box_h = y2 - y1
    len_ratio = 0.2
    d = min(box_w, box_h) * len_ratio
    corners = list()
    # top left
    corners.append([(x1, y1 + d), (x1, y1), (x1 + d, y1)])
    # top right
    corners.append([(x2 - d, y1), (x2, y1), (x2, y1 + d)])
    # bottom left
    corners.append([(x1, y2 - d), (x1, y2), (x1 + d, y2)])
    # bottom right
    corners.append([(x2 - d, y2), (x2, y2), (x2, y2 - d)])
    line_w = 2 if d * 0.4 > 2 else d * 0.4
    for corner in corners:
        (line_xs, line_ys) = zip(*corner)
        ax.add_line(Line2D(line_xs, line_ys,
                    linewidth=line_w, color=color_box))
        if text:
            ax.text(x1, y1 - 5, text, fontsize=6,
                    family='serif',
                    bbox=dict(facecolor=color_box, alpha=0.6, pad=0,
                            edgecolor='none'),
                    color='white')
    return ax


def vis_image_gvg(img, ceph_name, phrase , gt_objects, color_list, vis_dir, dataset, task_id,span_idx):
    fig = plt.figure() 
    ax_multi = fig.add_subplot(111)
    boxes_target = defaultdict(list)
    # sent, spans_str = cap.split('<&noun&>')
    # span_list = spans_str.split(';')
    if span_idx==0:
        gt_objects = gt_objects.strip().split(';')
    
    for i, label in enumerate(gt_objects):
        try:
            x0, y0, x1, y1 = label.strip().split(',')
        except:
            x0, y0, x1, y1 = label
        boxes_target["boxes"].append([float(x0), float(y0), float(x1), float(y1)])

    # for span_idx, span in enumerate(span_list):
    #     span_v = [int(x) for x in span.split(',')]
    #     phrase = sent[span_v[0]:span_v[1]]

    #     bbox_list = label_list[span_idx].split(',')
    #     bbox_list = [bbox_list[i: i+4] for i in range(0, len(bbox_list), 4) if i+4 <= len(bbox_list)]

    #     boxes_target = defaultdict(list)
        
    #     for ilabel in bbox_list:
    #         x0, y0, x1, y1 = ilabel
    #         boxes_target["boxes"].append([float(x0), float(y0), float(x1), float(y1)])

    box_alpha = 1
    span_idx=0
    for bbox_idx in range(len(boxes_target["boxes"])):
        in_phrase = phrase if bbox_idx == 0 else ''
        ax_multi = vis_bbox_with_text(
            boxes_target["boxes"][bbox_idx],  
            span_idx,
            in_phrase,
            ax_multi,
            color_list,
            box_alpha=box_alpha
        )
        span_idx+=1

        ax_multi.get_xaxis().set_visible(False)
        ax_multi.get_yaxis().set_visible(False)
        ax_multi.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        vis_image_name = '_'.join((task_id, ceph_name.split('/')[-1]))
        os.makedirs(os.path.join(vis_dir, dataset, task_id), exist_ok=True)
        # pdb.set_trace()
        fig.savefig(os.path.join(vis_dir, dataset, task_id, vis_image_name), bbox_inches='tight', pad_inches=0)
        plt.close('all')


color_list = colormap(rgb=True) / 255
# tsv_path1 = '/mnt/lustre/share_data/suyulin/VG/infer_bbox_self.tsv'
tsv_path2 = '/mnt/lustre/share_data/suyulin/VG/pseudo_tiny_semi_1/caption_0.tsv'
vis_dir = '/mnt/lustre/share_data/suyulin/VG/refcocog/vis_caption_0'
separator = '\t'


cnt = 0
cnt_dataset = defaultdict(int)
# f1_in=open(tsv_path1, 'r')
f2_in=open(tsv_path2, 'r')
# tsv1_reader = f1_in.readlines()#csv.reader(f1_in, delimiter=separator)
tsv2_reader = f2_in.readlines()#csv.reader(f2_in, delimiter=separator)
for line_idx in range(999):
# for line_idx, line in enumerate(tsv1_reader):
    
    # uniq_id,img_id, img_path, caption, refs, gt_objects, dataset_name, task_id = tsv1_reader[line_idx].split('\t')
    uniq_id,img_id, img_path, caption, refs, gt_objects, dataset_name, task_id = tsv2_reader[line_idx].split('\t')
    # except:
    #     continue
    # gt_objects=gt_objects+';'+gt_objects2
    file_name_bbox='/mnt/lustre/share_data/suyulin/VG/refcocog/detections_total/boxes.hdf5'
    
    f= h5py.File(file_name_bbox,'r')
    ref_objects=f[img_id][:]
    if cnt_dataset[dataset_name] < 100:
        cnt_dataset[dataset_name] += 1
        img = read_img_ceph(img_path)
        vis_image_gvg(img, img_path, refs, gt_objects, color_list, vis_dir, dataset_name, 'vg',0) 
        # vis_image_gvg(img, img_path, refs, ref_objects[:4], color_list, vis_dir, dataset_name, task_id,1)
        cnt += 1

    if cnt > 30000:
        break






            
