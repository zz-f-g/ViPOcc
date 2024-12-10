import argparse
import os
import sys
from collections import namedtuple
from pathlib import Path

import json
import torch
import tqdm
from PIL import Image

from snog_sampler import gen_snog_samples, draw

sys.path.append(os.path.join(os.getcwd(), "GroundingDINO"))
sys.path.append(os.path.join(os.getcwd(), "segment_anything"))

# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

# segment anything
from segment_anything import (
    sam_model_registry,
    SamPredictor
)
import cv2
import numpy as np

# a label and all meta information
Label = namedtuple('Label', [

    'name',  # The identifier of this label, e.g. 'car', 'person', ... .

    'color',  # The color of this label
])

# --------------------------------------------------------------------------------
# A list of all labels
# --------------------------------------------------------------------------------

# Please adapt the train IDs as appropriate for your approach.
# Note that you might want to ignore labels with ID 255 during training.
# Further note that the current train IDs are only a suggestion. You can use whatever you like.
# Make sure to provide your results using the original IDs and not the training IDs.
# Note that many IDs are ignored in evaluation and thus you never need to predict these!

labels = [
    #       name                     color
    Label('unlabeled', (0, 0, 0)),
    # Label(  'ego vehicle'          , (  0,  0,  0) ),
    # Label(  'rectification border' , (  0,  0,  0) ),
    # Label(  'out of roi'           , (  0,  0,  0) ),
    # Label(  'static'               , (  0,  0,  0) ),
    # Label(  'dynamic'              , (111, 74,  0) ),
    # Label(  'ground'               , ( 81,  0, 81) ),
    Label('road', (128, 64, 128)),
    # Label(  'sidewalk'             , (244, 35,232) ),
    # Label(  'parking'              , (250,170,160) ),
    # Label(  'rail track'           , (230,150,140) ),
    Label('building', (70, 70, 70)),
    # Label(  'wall'                 , (102,102,156) ),
    # Label(  'fence'                , (190,153,153) ),
    # Label(  'guard rail'           , (180,165,180) ),
    # Label(  'bridge'               , (150,100,100) ),
    # Label(  'tunnel'               , (150,120, 90) ),
    # Label(  'pole'                 , (153,153,153) ),
    # Label(  'polegroup'            , (153,153,153) ),
    # Label(  'traffic light'        , (250,170, 30) ),
    # Label(  'traffic sign'         , (220,220,  0) ),
    Label('vegetation', (107, 142, 35)),
    # Label(  'terrain'              , (152,251,152) ),
    Label('sky', (70, 130, 180)),
    # Label(  'person'               , (220, 20, 60) ),
    # Label(  'rider'                , (255,  0,  0) ),
    Label('car', (0, 0, 142)),
    # Label(  'truck'                , (  0,  0, 70) ),
    # Label(  'bus'                  , (  0, 60,100) ),
    # Label(  'caravan'              , (  0,  0, 90) ),
    # Label(  'trailer'              , (  0,  0,110) ),
    # Label(  'train'                , (  0, 80,100) ),
    # Label(  'motorcycle'           , (  0,  0,230) ),
    # Label(  'bicycle'              , (119, 11, 32) ),
    # Label(  'license plate'        , (  0,  0,142) ),
]

cs_phrases = '.'.join([label.name for label in labels[1:]])  # 'road.building.vegetation.sky.car...'
phrase_id_color = {label.name: (i, label.color) for i, label in enumerate(labels)}
palette = [label.color for label in labels]
palette = [c for rgb in palette for c in rgb]


def load_image(image_path):
    # load image
    image_pil = Image.open(image_path).convert("RGB")  # load image

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image


def load_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model


def get_grounding_output(model, image, caption, box_threshold, text_threshold, with_logits=True, device="cpu"):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
    logits.shape[0]

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
    logits_filt.shape[0]

    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    # build pred
    pred_phrases = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        if with_logits:
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        else:
            pred_phrases.append(pred_phrase)

    return boxes_filt, pred_phrases


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Grounded-SAM for KITTI-360")
    parser.add_argument(
        "--img_dir", type=str,
        default='YOUR/KITTI-360/PATH/data_2d_raw', help="path to image file"
    )
    parser.add_argument(
        "--config", type=str,
        default='GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py',
        help="path to config file")
    parser.add_argument(
        "--grounded_checkpoint", type=str,
        default='checkpoints/groundingdino_swint_ogc.pth',
        help="path to checkpoint file"
    )
    parser.add_argument("--sam_checkpoint", default='./checkpoints/sam_vit_h_4b8939.pth')
    parser.add_argument("--text_prompt", type=str, default=f'{cs_phrases}', help="text prompt")
    parser.add_argument("--box_threshold", type=float, default=0.3, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="text threshold")
    parser.add_argument("--device", type=str, default="cuda")

    # custom args
    parser.add_argument("--save_json", action="store_true", help="save json")
    parser.add_argument("--save_mask", action="store_true", help="save mask")

    parser.add_argument('--patch_size', type=int, default=8, help='size of the patch')
    parser.add_argument('--patch_num', type=int, default=64, help='number of patches')
    parser.add_argument('--gaussian_sample_phrases', default=['car', 'person'])
    parser.add_argument('--background_sample_ratio', type=float, default=0.5)
    parser.add_argument('--epochs', type=float, default=25)

    args = parser.parse_args()

    print('SAM path:', os.getcwd())

    # cfg
    config_file = args.config
    grounded_checkpoint = args.grounded_checkpoint
    sam_checkpoint = args.sam_checkpoint
    img_dir = args.img_dir
    text_prompt = args.text_prompt
    box_threshold = args.box_threshold
    text_threshold = args.text_threshold
    device = args.device
    save_json = args.save_json
    save_mask = args.save_mask

    # load model
    model = load_model(config_file, grounded_checkpoint, device=device)
    predictor = SamPredictor(sam_model_registry["vit_h"](checkpoint=sam_checkpoint).to(device))

    all_imgs = sorted(list(Path(img_dir).rglob('2013_05_28_drive_*_sync/image_*/data_192x640*/*.png')))

    for image_path in tqdm.tqdm(all_imgs):
        save_dir = Path(str(image_path).replace('data_2d_raw', 'samples')).parent
        save_dir.mkdir(parents=True, exist_ok=True)
        save_name = image_path.name
        save_sample_path = save_dir / save_name.replace('.png', '.npy')
        save_mask_path = save_dir / save_name
        save_mask_np_path = save_dir / save_name.replace('.png', '_mask.npy')
        save_json_path = save_dir / save_name.replace('.png', '.json')

        if save_mask_np_path.exists() and save_mask_path.exists() and save_json_path.exists():
            continue

        # load image
        image_pil, image = load_image(image_path)

        # run grounding dino model
        boxes_filt, pred_phrases = get_grounding_output(
            model, image, text_prompt, box_threshold, text_threshold, device=device)

        image = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
        predictor.set_image(image)

        if len(boxes_filt) == 0:  # no detections from grounding dino
            print('No boxes detected in {}.'.format(image_path))
            mask_img = np.zeros(image.shape[:2], dtype=np.uint8)  # [192, 640]
            # np.save(save_np_path, mask_img)
            mask_img = Image.fromarray(mask_img, mode='P')
            mask_img.putpalette(palette)
            mask_img.convert('P', palette=Image.Palette.ADAPTIVE, colors=len(palette))
            mask_img = mask_img.convert('RGB')
            mask_img.save(save_mask_path)
            continue

        size = image_pil.size
        H, W = size[1], size[0]
        for i in range(boxes_filt.size(0)):
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            boxes_filt[i][2:] += boxes_filt[i][:2]

        transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(device)

        masks, _, _ = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )

        mask_img = np.zeros(masks.shape[-2:], dtype=np.uint8)  # [192, 640]
        json_data = []
        for idx, (phrase, mask, box) in enumerate(zip(pred_phrases, masks, boxes_filt), start=1):
            mask = mask.cpu().numpy()[0]
            phrase, logit = phrase.split('(')
            logit = logit[:-1]  # remove `)`
            if ' ' in phrase:
                phrase = phrase.split(' ')[0]
            phrase = phrase.strip().replace('.', '')  # 'road', 'car', ...
            pixel_num = mask.sum()
            json_data.append({
                'label': phrase,
                'logit': float(logit),
                'box': box.numpy().tolist(),
                'pixel_num': float(pixel_num)
            })
            mask_img[mask] = phrase_id_color[phrase][0]
        if save_json:
            with open(save_json_path, 'w') as f:
                json.dump(json_data, f)
        if save_mask:
            np.save(save_mask_np_path, mask_img)
            mask = Image.fromarray(mask_img, mode='P')
            mask.putpalette(palette)
            mask.convert('P', palette=Image.Palette.ADAPTIVE, colors=len(palette))
            mask = mask.convert('RGB')
            mask.save(save_mask_path)

        # generate samples using snog sampler
        all_samples = []
        for epoch in range(args.epochs):
            snog_samples = gen_snog_samples(json=json_data,
                                            img_size=size,
                                            patch_size=args.patch_size,
                                            patch_num=args.patch_num,
                                            background_sample_ratio=args.background_sample_ratio,
                                            gaussian_sample_phrases=args.gaussian_sample_phrases
                                            )

            all_samples.append(snog_samples)
        all_samples = np.stack(all_samples).astype(np.uint16)
        np.save(save_sample_path, all_samples)  # [epochs, num_anchors, 2]
