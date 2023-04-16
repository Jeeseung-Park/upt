import argparse
import os
import sys
sys.path.append('detr')
from models import build_model
import torch
import random
import warnings
import argparse
import numpy as np
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler

from upt import build_detector
from utils import custom_collate, CustomisedDLE, DataFactory
from util.misc import nested_tensor_from_tensor_list

from tqdm import tqdm
import pocket
import json



def main(args):
    
    device = torch.device("cuda:0")
    testset = DataFactory(name=args.dataset, partition=args.partitions[1], data_root=args.data_root)
    test_loader = DataLoader(
        dataset=testset,
        collate_fn=custom_collate, batch_size=1,
        num_workers=args.num_workers, pin_memory=True, drop_last=False,
        sampler=torch.utils.data.SequentialSampler(testset)
    )
    
    args.human_idx = 0
    object_to_target = test_loader.dataset.dataset.object_to_verb
    args.num_classes = 117
    
    detr, _, postprocessors = build_model(args)
    postprocessor = postprocessors['bbox']
    detr.load_state_dict(torch.load(args.pretrained, map_location='cpu')['model_state_dict'])
    detr.eval()
    detr.to(device)
    
    dataset = test_loader.dataset.dataset
    
    with open("hicodet/instances_test2015.json") as f:
        upt_json = json.load(f)
    with open(os.path.join(args.base_dir, "hicodet/instances_test2015.json")) as f:
        scg_json = json.load(f)
    
    upt_to_scg_object_idx = list(map(lambda x: scg_json['objects'].index(x), upt_json['objects']))
    
    with torch.no_grad():
        for idx, batch in tqdm(enumerate(test_loader)):
            images = pocket.ops.relocate_to_cuda(batch[0])
            
            image_sizes = torch.as_tensor([
                im.size()[-2:] for im in images
            ], device=images[0].device)
            
            if isinstance(images, (list, torch.Tensor)):
                images = nested_tensor_from_tensor_list(images)
            
            features, pos = detr.backbone(images)

            src, mask = features[-1].decompose()
            assert mask is not None
            hs = detr.transformer(detr.input_proj(src), mask, detr.query_embed.weight, pos[-1])[0]

            outputs_class = detr.class_embed(hs)
            outputs_coord = detr.bbox_embed(hs).sigmoid()

            results = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
            results = postprocessor(results, image_sizes)
            #print(results)
            assert len(results) == 1
            result = pocket.ops.relocate_to_cpu(results[0], ignore=True)
            image_id = dataset._idx[idx]
            #print(image_id)
            boxes = result['boxes']
            labels = result['labels']
            scores = result['scores']
            ow, oh = dataset.image_size(idx)
            h, w = image_sizes[0]
            scale_fct = torch.as_tensor([
                    ow / w, oh / h, ow / w, oh / h
                ]).unsqueeze(0)
            
            boxes *= scale_fct
            
            boxes[:,0] = boxes[:,0].clip(0, ow)
            boxes[:,1] = boxes[:,1].clip(0, oh)
            boxes[:,2] = boxes[:,2].clip(0, ow)
            boxes[:,3] = boxes[:,3].clip(0, oh)
            
            sorted_idx = scores.argsort(descending=True)
            keep_idx = (scores[sorted_idx].numpy()>0.05).nonzero()[0]
            det_json = {}
            det_json['boxes'] = boxes[sorted_idx].numpy()[keep_idx].tolist()
            upt_labels = labels[sorted_idx].numpy()[keep_idx].tolist()
            scg_labels = list(map(lambda x: upt_to_scg_object_idx[x], upt_labels))
            det_json['labels'] = scg_labels
            det_json['scores'] = scores[sorted_idx].numpy()[keep_idx].tolist()
            
            #print(det_json)

            os.makedirs(os.path.join(args.base_dir, "hicodet/detections/test2015_upt/"), exist_ok=True)
            
            with open(os.path.join(args.base_dir, f"hicodet/detections/test2015_upt/{dataset._filenames[image_id].replace('jpg', 'json')}"), "w") as f:
                json.dump(det_json, f)
        




if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--base_dir", default="../VIPLO", type=str)
    
    parser.add_argument('--lr-head', default=1e-4, type=float)
    parser.add_argument('--batch-size', default=2, type=int)
    parser.add_argument('--weight-decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--lr-drop', default=10, type=int)
    parser.add_argument('--clip-max-norm', default=0.1, type=float)

    parser.add_argument('--backbone', default='resnet50', type=str)
    parser.add_argument('--dilation', action='store_true')
    parser.add_argument('--position-embedding', default='sine', type=str, choices=('sine', 'learned'))

    parser.add_argument('--repr-dim', default=512, type=int)
    parser.add_argument('--hidden-dim', default=256, type=int)
    parser.add_argument('--enc-layers', default=6, type=int)
    parser.add_argument('--dec-layers', default=6, type=int)
    parser.add_argument('--dim-feedforward', default=2048, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--nheads', default=8, type=int)
    parser.add_argument('--num-queries', default=100, type=int)
    parser.add_argument('--pre-norm', action='store_true')

    parser.add_argument('--no-aux-loss', dest='aux_loss', action='store_false')
    parser.add_argument('--set-cost-class', default=1, type=float)
    parser.add_argument('--set-cost-bbox', default=5, type=float)
    parser.add_argument('--set-cost-giou', default=2, type=float)
    parser.add_argument('--bbox-loss-coef', default=5, type=float)
    parser.add_argument('--giou-loss-coef', default=2, type=float)
    parser.add_argument('--eos-coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    parser.add_argument('--alpha', default=0.5, type=float)
    parser.add_argument('--gamma', default=0.2, type=float)

    parser.add_argument('--dataset', default='hicodet', type=str)
    parser.add_argument('--partitions', nargs='+', default=['train2015', 'test2015'], type=str)
    parser.add_argument('--num-workers', default=2, type=int)
    parser.add_argument('--data-root', default='./hicodet')

    # training parameters
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--port', default='1234', type=str)
    parser.add_argument('--seed', default=66, type=int)
    parser.add_argument('--pretrained', default='', help='Path to a pretrained detector')
    parser.add_argument('--resume', default='', help='Resume from a model')
    parser.add_argument('--output-dir', default='checkpoints')
    parser.add_argument('--print-interval', default=500, type=int)
    parser.add_argument('--world-size', default=1, type=int)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--cache', action='store_true')
    parser.add_argument('--sanity', action='store_true')
    parser.add_argument('--box-score-thresh', default=0.2, type=float)
    parser.add_argument('--fg-iou-thresh', default=0.5, type=float)
    parser.add_argument('--min-instances', default=3, type=int)
    parser.add_argument('--max-instances', default=15, type=int)

    args = parser.parse_args()
    print(args)
    with open("parser.json", "w") as f:
        json.dump(args.__dict__, f)
    main(args)