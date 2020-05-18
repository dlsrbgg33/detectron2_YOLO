# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import copy
import logging
import numpy as np
import torch
from fvcore.common.file_io import PathManager
from PIL import Image

from . import detection_utils as utils
from . import transforms as T
import random
from detectron2.utils.labels_viper16 import id2label, trainId2label
from detectron2.utils.labels_city import id2label_r, trainId2label_r
import glob
import os
"""
This file contains the default mapping that's applied to "dataset dicts".
"""

__all__ = ["DatasetMapper", "DatasetMapper_test" ,"DatasetMapper_SEG"]


class DatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by the model.

    This is the default callable to be used to map your dataset dict into training data.
    You may need to follow it to implement your own one for customized logic,
    such as a different way to read or transform images.
    See :doc:`/tutorials/data_loading` for details.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies cropping/geometric transforms to the image and annotations
    3. Prepare data and annotations to Tensor and :class:`Instances`
    """
    # import pdb
    # pdb.set_trace()
    def __init__(self, cfg, video, is_train=True):
        if cfg.INPUT.CROP.ENABLED and is_train:
            self.crop_gen = T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE)
            logging.getLogger(__name__).info("CropGen used in training: " + str(self.crop_gen))
        else:
            self.crop_gen = None

        self.tfm_gens = utils.build_transform_gen(cfg, is_train)
        # import pdb
        # pdb.set_trace()
        # fmt: off
        self.img_format     = cfg.INPUT.FORMAT
        self.mask_on        = cfg.MODEL.MASK_ON
        self.mask_format    = cfg.INPUT.MASK_FORMAT
        self.keypoint_on    = cfg.MODEL.KEYPOINT_ON
        self.load_proposals = cfg.MODEL.LOAD_PROPOSALS

        self.label_viper    = cfg.MODEL.LABEL_CON_VIPER
        # fmt: on
        # import pdb
        # pdb.set_trace()
        if self.keypoint_on and is_train:
            # Flip only makes sense in training
            self.keypoint_hflip_indices = utils.create_keypoint_hflip_indices(cfg.DATASETS.TRAIN)
        else:
            self.keypoint_hflip_indices = None

        if self.load_proposals:
            self.min_box_side_len = cfg.MODEL.PROPOSAL_GENERATOR.MIN_SIZE
            self.proposal_topk = (
                cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TRAIN
                if is_train
                else cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TEST
            )

        self.label_2_id = 255 * np.ones((256,))
        for l in id2label:
            if l in (-1, 255):
                continue
            self.label_2_id[l] = id2label[l].trainId

        self.is_train = is_train
        self.video = video
        self.video_inter = cfg.DATASETS.INTERV_VIDEO
        self.video_randrage = cfg.DATASETS.VID_CHOICE == 'radn_range'

    def __call__(self, dataset_dict):
    # def forward(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        ### random sampling ###

        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # USER: Write your own image loading if it's not from a file
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        if self.video:
            if self.video_randrage:
                up_interv = dataset_dict["fle_idx"] + random.randrange(1,self.video_inter)
                bt_interv = dataset_dict["fle_idx"] - random.randrange(1,self.video_inter)
            else:
                up_interv = dataset_dict["fle_idx"] + self.video_inter
                bt_interv = dataset_dict["fle_idx"] - self.video_inter
            
            ref_idx = '%03d_%05d.jpg' %(dataset_dict["fld_idx"], up_interv) if up_interv <= dataset_dict["length"] else \
                      '%03d_%05d.jpg' %(dataset_dict["fld_idx"], bt_interv)
            ref_idx_gt = '%03d_%05d.png' %(dataset_dict["fld_idx"], up_interv) if up_interv <= dataset_dict["length"] else \
                      '%03d_%05d.png' %(dataset_dict["fld_idx"], bt_interv)

            ref_img = utils.read_image('/'.join(dataset_dict["file_name"].split('/')[:-1] + [ref_idx]), format=self.img_format)
            utils.check_image_size(dataset_dict, ref_img)
        utils.check_image_size(dataset_dict, image)
        
        if "annotations" not in dataset_dict:
            # if self.video:
            image, ref_img, transforms = T.apply_transform_gens(
                ([self.crop_gen] if self.crop_gen else []) + self.tfm_gens, image, ref_img
            ) if self.video is True else T.apply_transform_gens(
            ([self.crop_gen] if self.crop_gen else []) + self.tfm_gens, image, None
            )
        #### 1.RandomResize 2. RandomCrop [800, 1600]
        else:
            # Crop around an instance if there are instances in the image.
            # USER: Remove if you don't use cropping
            image, ref_img, transforms = T.apply_transform_gens(self.tfm_gens, image, ref_img) if self.video is True else \
                                T.apply_transform_gens(self.tfm_gens, image, None)
            if self.crop_gen:
                crop_tfm = utils.gen_crop_transform_with_instance(
                    self.crop_gen.get_crop_size(image.shape[:2]),
                    image.shape[:2],
                    np.random.choice(dataset_dict["annotations"]),
                )
                image = crop_tfm.apply_image(image)
                if self.video:
                    ref_img = crop_tfm.apply_image(ref_img)

            if self.crop_gen:
                transforms = transforms + crop_tfm

        image_shape = image.shape[:2]  # h, w
        if ref_img is not None:
            ref_image_shape = ref_img.shape[:2]  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        if self.video:
            dataset_dict["ref_image"] = torch.as_tensor(np.ascontiguousarray(ref_img.transpose(2, 0, 1)))

        # USER: Remove if you don't use pre-computed proposals.
        if self.load_proposals:
            utils.transform_proposals(
                dataset_dict, image_shape, transforms, self.min_box_side_len, self.proposal_topk
            )

        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                if not self.mask_on:
                    anno.pop("segmentation", None)
                if not self.keypoint_on:
                    anno.pop("keypoints", None)

            # USER: Implement additional transformations if you have other types of data
            annos = [
                utils.transform_instance_annotations(
                    obj, transforms, image_shape, keypoint_hflip_indices=self.keypoint_hflip_indices
                )
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = utils.annotations_to_instances(
                annos, image_shape, mask_format=self.mask_format
            )
            # Create a tight bounding box from masks, useful when image is cropped
            if self.crop_gen and instances.has("gt_masks"):
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            dataset_dict["instances"] = utils.filter_empty_instances(instances)

        # USER: Remove if you don't do semantic/panoptic segmentation.
        


        if "sem_seg_file_name" in dataset_dict:
            dataset_dict_copy = copy.deepcopy(dataset_dict)
            with PathManager.open(dataset_dict.pop("sem_seg_file_name"), "rb") as f:
                sem_seg_gt = Image.open(f)
                sem_seg_gt = np.asarray(sem_seg_gt, dtype="uint8")
                if self.video:
                    sem_seg_ref_gt = Image.open('/'.join(dataset_dict_copy["sem_seg_file_name"].split('/')[:-1] + [ref_idx_gt]))
                    sem_seg_ref_gt = np.asarray(sem_seg_ref_gt, dtype="uint8")
            if self.label_viper:
                sem_seg_gt = self.label_2_id[sem_seg_gt]
                if self.video:
                    sem_seg_ref_gt = self.label_2_id[sem_seg_ref_gt]

            sem_seg_gt = transforms.apply_segmentation(sem_seg_gt)
            sem_seg_gt = torch.as_tensor(sem_seg_gt.astype("long"))
            dataset_dict["sem_seg"] = sem_seg_gt
            if self.video:
                sem_seg_ref_gt = transforms.apply_segmentation(sem_seg_ref_gt)
                sem_seg_ref_gt = torch.as_tensor(sem_seg_ref_gt.astype("long"))
                dataset_dict["sem_seg_ref"] = sem_seg_ref_gt

        return dataset_dict

class DatasetMapper_test:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by the model.

    This is the default callable to be used to map your dataset dict into training data.
    You may need to follow it to implement your own one for customized logic,
    such as a different way to read or transform images.
    See :doc:`/tutorials/data_loading` for details.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies cropping/geometric transforms to the image and annotations
    3. Prepare data and annotations to Tensor and :class:`Instances`
    """
    # import pdb
    # pdb.set_trace()
    def __init__(self, cfg, video, is_train=True):
        if cfg.INPUT.CROP.ENABLED and is_train:
            self.crop_gen = T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE)
            logging.getLogger(__name__).info("CropGen used in training: " + str(self.crop_gen))
        else:
            self.crop_gen = None

        self.tfm_gens = utils.build_transform_gen(cfg, is_train)
        # import pdb
        # pdb.set_trace()
        # fmt: off
        self.img_format     = cfg.INPUT.FORMAT
        self.mask_on        = cfg.MODEL.MASK_ON
        self.mask_format    = cfg.INPUT.MASK_FORMAT
        self.keypoint_on    = cfg.MODEL.KEYPOINT_ON
        self.load_proposals = cfg.MODEL.LOAD_PROPOSALS

        self.label_viper    = cfg.MODEL.LABEL_CON_VIPER
        # fmt: on

        self.label_2_id = 255 * np.ones((256,))
        for l in id2label:
            if l in (-1, 255):
                continue
            self.label_2_id[l] = id2label[l].trainId

        self.is_train = is_train
        self.video = video
        self.reverse_ref = cfg.DATASETS.REV_TEST
        self.interF_train = cfg.DATASETS.INTERF_TRAIN
        self.interF_test  = cfg.DATASETS.INTERF_TEST
        self.video_randrage = cfg.DATASETS.VID_CHOICE == 'radn_range'

    def __call__(self, dataset_dict):
    # def forward(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        ### random sampling ###

        if len(dataset_dict['file_name'].split('/')) == 6:
            viper_data = True
        else:
            viper_data = False

        flip = np.random.choice(2) * 2 -1
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # USER: Write your own image loading if it's not from a file
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        if self.video:
            if self.is_train:
                if self.video_randrage:
                    vid_interv_top = dataset_dict["fle_idx"] + flip * random.randrange(1,self.interF_train)
                    vid_interv_bot = dataset_dict["fle_idx"] - flip * random.randrange(1,self.interF_train)
                else:
                    vid_interv_top = dataset_dict["fle_idx"] + flip * self.interF_train
                    vid_interv_bot = dataset_dict["fle_idx"] - flip * self.interF_train
                
                if viper_data:
                    ref_idx_ = '%03d_%05d.jpg' %(dataset_dict["fld_idx"], vid_interv_top) if vid_interv_top <= dataset_dict["length"] and \
                                vid_interv_top > 0 else \
                              '%03d_%05d.jpg' %(dataset_dict["fld_idx"], vid_interv_bot)
                else:
                    ref_idx = '%06d_%06d_leftImg8bit.png' %(dataset_dict["fld_idx"], vid_interv_top) if vid_interv_top < dataset_dict["length"] and \
                                vid_interv_top >= 0 else \
                              '%06d_%06d_leftImg8bit.png' %(dataset_dict["fld_idx"], vid_interv_bot)
                    ref_idx_ = dataset_dict["city"] + '_' + ref_idx
            else:
                ref_idx_file = dataset_dict["fle_idx"] - self.interF_test
                #### if ref_idx is not bigger than 0, main image is not used in evaluation.
                if ref_idx_file <= 0:
                    if not self.reverse_ref:
                        ref_idx_ = '%03d_%05d.jpg' %(dataset_dict["fld_idx"], dataset_dict["fle_idx"])
                    else:
                        ref_idx_file = dataset_dict["fle_idx"] + self.interF_test
                        ref_idx_ = '%03d_%05d.jpg' %(dataset_dict["fld_idx"], ref_idx_file)
                        
                #### else, main image should be supported from ref image
                else:
                    if viper_data:
                        ref_idx_ = '%03d_%05d.jpg' %(dataset_dict["fld_idx"], ref_idx_file)
                    else:
                        ref_idx = '%06d_%06d_leftImg8bit.png' %(dataset_dict["fld_idx"], ref_idx_file)
                        ref_idx_ = dataset_dict["city"] + '_' + ref_idx


            ref_idx = os.path.join('/'.join(dataset_dict["file_name"].split('/')[:-1]), ref_idx_)
            ref_img = utils.read_image(ref_idx, format=self.img_format)
            utils.check_image_size(dataset_dict, ref_img)
        utils.check_image_size(dataset_dict, image)

        # if self.video:
        image, ref_img, transforms = T.apply_transform_gens(
            ([self.crop_gen] if self.crop_gen else []) + self.tfm_gens, image, ref_img
        ) if self.video is True else T.apply_transform_gens(
        ([self.crop_gen] if self.crop_gen else []) + self.tfm_gens, image, None
        )
        #### 1.RandomResize 2. RandomCrop [800, 1600]

        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        if self.video:
            dataset_dict["ref_image"] = torch.as_tensor(np.ascontiguousarray(ref_img.transpose(2, 0, 1)))
            dataset_dict["ref_file_name"] = ref_idx
        ##### Only viper
        if viper_data:
            suffix = 'jpg'
            if "sem_seg_file_name" in dataset_dict:
                dataset_dict_copy = copy.deepcopy(dataset_dict)
                with PathManager.open(dataset_dict.pop("sem_seg_file_name"), "rb") as f:
                    sem_seg_gt = Image.open(f)
                    sem_seg_gt = np.asarray(sem_seg_gt, dtype="uint8")
                    if self.video:
                        ref_idx_ = ref_idx_[:-len(suffix)] + 'png'
                        sem_seg_ref_gt = Image.open(os.path.join('/'.join(dataset_dict_copy["sem_seg_file_name"].split('/')[:-1]), ref_idx_))
                        sem_seg_ref_gt = np.asarray(sem_seg_ref_gt, dtype="uint8")
                if self.label_viper:
                    sem_seg_gt = self.label_2_id[sem_seg_gt]
                    if self.video:
                        sem_seg_ref_gt = self.label_2_id[sem_seg_ref_gt]

                sem_seg_gt = transforms.apply_segmentation(sem_seg_gt)
                sem_seg_gt = torch.as_tensor(sem_seg_gt.astype("long"))
                dataset_dict["sem_seg"] = sem_seg_gt
                if self.video:
                    sem_seg_ref_gt = transforms.apply_segmentation(sem_seg_ref_gt)
                    sem_seg_ref_gt = torch.as_tensor(sem_seg_ref_gt.astype("long"))
                    dataset_dict["sem_seg_ref"] = sem_seg_ref_gt

        return dataset_dict


class DatasetMapper_SEG:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by the model.

    This is the default callable to be used to map your dataset dict into training data.
    You may need to follow it to implement your own one for customized logic,
    such as a different way to read or transform images.
    See :doc:`/tutorials/data_loading` for details.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies cropping/geometric transforms to the image and annotations
    3. Prepare data and annotations to Tensor and :class:`Instances`
    """

    def __init__(self, cfg, video, is_train=True):
        if cfg.INPUT.CROP.ENABLED and is_train:
            self.crop_gen = T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE)
            logging.getLogger(__name__).info("CropGen used in training: " + str(self.crop_gen))
        else:
            self.crop_gen = None

        self.tfm_gens = utils.build_transform_gen(cfg, is_train)
        # import pdb
        # pdb.set_trace()
        # fmt: off
        self.img_format     = cfg.INPUT.FORMAT
        self.mask_on        = cfg.MODEL.MASK_ON
        self.mask_format    = cfg.INPUT.MASK_FORMAT
        self.keypoint_on    = cfg.MODEL.KEYPOINT_ON
        self.load_proposals = cfg.MODEL.LOAD_PROPOSALS

        self.label_viper    = cfg.MODEL.LABEL_CON_VIPER
        # fmt: on

        self.label_2_id = 255 * np.ones((256,))
        for l in id2label:
            if l in (-1, 255):
                continue
            self.label_2_id[l] = id2label[l].trainId

        self.is_train = is_train
        self.video = video
        self.video_inter = cfg.DATASETS.INTERV_VIDEO
        self.video_randrage = cfg.DATASETS.VID_CHOICE == 'radn_range'

    def __call__(self, dataset_dict):
    # def forward(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        ### random sampling ###


        

        #### Determine where it comes from
        if len(dataset_dict['file_name'].split('/')) == 6:
            viper_data = True
        else:
            viper_data = False

        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # USER: Write your own image loading if it's not from a file
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        if self.video:
            if self.video_randrage:
                video_interv = random.randrange(1,self.video_inter)
            else:
                video_interv = self.video_inter
            flip = np.random.choice(2) * 2 -1
            
            if viper_data:
                ref_idx_pre = '/'.join(dataset_dict['file_name'].split('/')[:-1])
                ref_idx_ = str(int(dataset_dict['file_name'].split('.')[0].split('_')[-1]) + flip * video_interv).zfill(5)
                ref_idx = ref_idx_ \
                if int(ref_idx_) > 0 and int(ref_idx_) <= len(glob.glob('/'.join(dataset_dict['file_name'].split('/')[:-1]) + '/*')) \
                else str(int(dataset_dict['file_name'].split('.')[0].split('_')[1]) - flip * video_interv)
                ref_final_idx_ = '/' + ref_idx_pre.split('/')[-1] + '_' + ref_idx
                ref_final_idx  = ref_final_idx_ + '.jpg'
            else:
                suffix = "_leftImg8bit.png"
                suffix_pre = 6
                ref_idx_pre = dataset_dict['file_name'][:-len(suffix)] 
                ref_idx_pre = ref_idx_pre[:-suffix_pre]
                ref_idx_ = str(int(dataset_dict['file_name'].split('/')[-1][:-len(suffix)].split('_')[-1]) + flip * video_interv).zfill(6)
                ref_idx =  ref_idx_\
                if int(ref_idx_) > 0 and int(ref_idx_) <= 30 \
                else str(int(ref_idx_pre.split('/')[-1].split('_')[1]) - flip * video_interv)
                ref_final_idx = ref_idx + suffix

            ref_img = utils.read_image(ref_idx_pre + ref_final_idx, format=self.img_format)

            utils.check_image_size(dataset_dict, ref_img)
        utils.check_image_size(dataset_dict, image)


        image, ref_img, transforms = T.apply_transform_gens(
            ([self.crop_gen] if self.crop_gen else []) + self.tfm_gens, image, ref_img
        ) if self.video is True else T.apply_transform_gens(
        ([self.crop_gen] if self.crop_gen else []) + self.tfm_gens, image, None
        )

        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        if self.video:
            dataset_dict["ref_image"] = torch.as_tensor(np.ascontiguousarray(ref_img.transpose(2, 0, 1)))

        ########################## SEG ########################
        dataset_dict_copy = copy.deepcopy(dataset_dict)
        # with PathManager.open(dataset_dict.pop("sem_seg_file_name"), "rb") as f:
        if viper_data:
            f = dataset_dict["sem_seg_file_name"]
            sem_seg_gt = Image.open(f)
            sem_seg_gt = np.asarray(sem_seg_gt, dtype="uint8")
        if self.video:
            if viper_data:
                sem_seg_ref_gt = '/'.join(f.split('/')[:-1])  + ref_final_idx_ + '.png'
                sem_seg_ref_gt = Image.open(sem_seg_ref_gt)
                sem_seg_ref_gt = np.asarray(sem_seg_ref_gt, dtype="uint8")
                sem_seg_ref_gt = self.label_2_id[sem_seg_ref_gt]
                 
                sem_seg_gt = self.label_2_id[sem_seg_gt]

                sem_seg_gt = transforms.apply_segmentation(sem_seg_gt)
                sem_seg_gt = torch.as_tensor(sem_seg_gt.astype("long"))
                dataset_dict["sem_seg"] = sem_seg_gt

                if self.video:
                    sem_seg_ref_gt = transforms.apply_segmentation(sem_seg_ref_gt)
                    sem_seg_ref_gt = torch.as_tensor(sem_seg_ref_gt.astype("long"))
                    dataset_dict["sem_seg_ref"] = sem_seg_ref_gt
        else:
            if viper_data:
                sem_seg_gt = self.label_2_id[sem_seg_gt]
            sem_seg_gt = transforms.apply_segmentation(sem_seg_gt)
            sem_seg_gt = torch.as_tensor(sem_seg_gt.astype("long"))
            dataset_dict["sem_seg"] = sem_seg_gt


        return dataset_dict
