import json
import pickle
import torch
import math
import h5py
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate


def invert_dict(d):
    return {v: k for k, v in d.items()}


def load_vocab(path):
    with open(path, 'r') as f:
        vocab = json.load(f)
        vocab['question_idx_to_token'] = invert_dict(vocab['question_token_to_idx'])
        vocab['answer_idx_to_token'] = invert_dict(vocab['answer_token_to_idx'])
    return vocab


def load_glove_embeddings(path, model, device):
    with open(path, 'rb') as glovef:
        glove_matrix = pickle.load(glovef)

    glove_matrix = torch.FloatTensor(glove_matrix).to(device)
    with torch.no_grad():
        model.token_embedding.weight.set_(glove_matrix)
    return model


def init_dataloader(data_label, loader_kwargs):
    if data_label == 'VQA2':
        DataLoader = VQADataLoader
        num_answers_per_q = 10
    elif 'perturb' in data_label:
        DataLoader = PerturbVQADataLoader
        num_answers_per_q = 10
    else:
        raise ValueError('Unrecognized dataset flag')

    return DataLoader(**loader_kwargs), num_answers_per_q, data_label


def init_perturb_dataloaders(loader_kwargs, get_iters=True):
    '''
    Loads dataset indexing information for perturbation subsets.
    The keyword argument 'perturbations' must be a list or None.
    The returned dictionaries have one dataloader per subset.
    '''
    data_label = 'perturb'

    perturb_list = loader_kwargs.pop('perturbations', None)
    if not perturb_list:
        if get_iters:
            return {}, -1, data_label, {}
        else:
            return {}, -1, data_label

    perturb_loaders_ = {}
    num_answers_per_q = -1

    for pert in perturb_list:
        loader_kwargs['perturb_type'] = pert
        d_loader, num_answers_per_q, _ = init_dataloader(data_label, loader_kwargs)
        perturb_loaders_[pert] = d_loader

    if get_iters:
        perturb_loader_iters_ = {k: iter(p_l) for k, p_l in perturb_loaders_.items()}
        return perturb_loaders_, num_answers_per_q, data_label, perturb_loader_iters_
    else:
        return perturb_loaders_, num_answers_per_q, data_label


def load_id_data(id_info_fname, subset_type):
    if subset_type:
        if isinstance(subset_type, str):
            subset_type = [subset_type]

    with open(id_info_fname, 'r') as idf:
        id_info = json.load(idf)

    q_image_ids, q_idxs, og_q_idxs = [], [], []

    if subset_type:
        for subt in subset_type:
            q_image_ids += id_info['perturb_image_ids'][subt]
            q_idxs += id_info['perturb_q_idxs'][subt]
            og_q_idxs += id_info['original_q_idxs'][subt]
    else:
        q_image_ids = id_info['image_ids']
        q_idxs = IdentityIndexer()
        og_q_idxs = IdentityIndexer()

    return q_image_ids, q_idxs, og_q_idxs


class VQADataset(Dataset):
    def __init__(self, question_h5_file, feat_coco_id_to_index, q_to_image_ids, q_to_idx, feature_path,
                 num_answer, use_spatial, get_boxes=False):
        self.question_h5_file = question_h5_file
        self.q_to_image_ids = q_to_image_ids
        self.q_to_idx = q_to_idx

        self.feature_path = feature_path
        self.feat_coco_id_to_index = feat_coco_id_to_index
        self.num_answer = num_answer
        self.use_spatial = use_spatial
        self.get_boxes = get_boxes

    def _fetch_question_data(self, q_index):
        with h5py.File(self.question_h5_file, 'r') as f:
            question = f['questions'][q_index]
            question_len = f['questions_len'][q_index]
            answer = f['answers'][q_index]

            if answer is not None:
                _answer = torch.zeros(self.num_answer)
                for i in answer:
                    if i == -1:
                        continue
                    _answer[i] += 1
                answer = _answer

            question = torch.LongTensor(question)
            question_len = torch.LongTensor(question_len)

        return question, question_len, answer

    def _fetch_image_data(self, image_id):
        feature_h5_fn = self.feature_path
        with h5py.File(feature_h5_fn, 'r') as f:
            image_index = self.feat_coco_id_to_index[image_id]
            vision_feat = f['features'][image_index]
            boxes = f['boxes'][image_index]
            w = f['widths'][image_index]
            h = f['heights'][image_index]

        if self.use_spatial:
            spatial_feat = np.zeros((5, len(boxes[0])))
            spatial_feat[0, :] = boxes[0, :] * 2 / w - 1  # x1
            spatial_feat[1, :] = boxes[1, :] * 2 / h - 1  # y1
            spatial_feat[2, :] = boxes[2, :] * 2 / w - 1  # x2
            spatial_feat[3, :] = boxes[3, :] * 2 / h - 1  # y2
            spatial_feat[4, :] = (spatial_feat[2, :] - spatial_feat[0, :]) * (spatial_feat[3, :] - spatial_feat[1, :])

            vision_feat = np.concatenate((vision_feat, spatial_feat), axis=0)

        vision_feat = torch.from_numpy(vision_feat).float()

        num_feat = boxes.shape[1]
        relation_mask = np.zeros((num_feat, num_feat))
        for i in range(num_feat):  # only need to set values for existing objects
            for j in range(i + 1, num_feat):
                # if there is no overlap between two bounding box
                if (boxes[0, i] >= boxes[2, j]
                        or boxes[0, j] >= boxes[2, i]
                        or boxes[1, i] >= boxes[3, j]
                        or boxes[1, j] >= boxes[3, i]):
                    pass
                else:
                    relation_mask[i, j] = relation_mask[j, i] = 1

        relation_mask = torch.from_numpy(relation_mask).byte()

        return vision_feat, relation_mask, boxes

    def __getitem__(self, index):
        q_idx = self.q_to_idx[index]
        q_data = self._fetch_question_data(q_idx)

        image_idx = self.q_to_image_ids[index]
        vision_feat, relation_mask, boxes = self._fetch_image_data(image_idx)

        question, question_len, answer = q_data
        example_ = [
            image_idx,
            answer,
            question,
            question_len[0],
            vision_feat,
            relation_mask
        ]

        if self.get_boxes:
            example_.append(boxes)

        example_ = tuple(example_)

        return example_

    def __len__(self):
        return len(self.q_to_image_ids)


class PerturbVQADataset(VQADataset):
    def __init__(self,
                 question_h5_file,
                 feat_coco_id_to_index,
                 q_to_image_ids,
                 q_to_idx,
                 q_to_og_idx,
                 feature_path,
                 num_answer,
                 use_spatial,
                 get_boxes=False):
        super().__init__(question_h5_file,
                         feat_coco_id_to_index,
                         q_to_image_ids,
                         q_to_idx,
                         feature_path,
                         num_answer,
                         use_spatial,
                         get_boxes)

        self.q_to_og_idx = q_to_og_idx

    def __getitem__(self, index):
        q_idx = self.q_to_idx[index]
        question, question_len, answer = self._fetch_question_data(q_idx)

        og_q_idx = self.q_to_og_idx[index]
        og_question, og_question_len, og_answer = self._fetch_question_data(og_q_idx)

        image_idx = self.q_to_image_ids[index]
        vision_feat, relation_mask, boxes = self._fetch_image_data(image_idx)

        example_ = [
            image_idx,
            answer,
            question,
            question_len[0],
            vision_feat,
            relation_mask,
            og_answer,
            og_question,
            og_question_len[0]
        ]

        if self.get_boxes:
            example_.append(boxes)

        example_ = tuple(example_)

        return example_


class VQADataLoader(DataLoader):
    def __init__(self, **kwargs):
        vocab_json_path = str(kwargs.pop('vocab_json'))
        print('loading vocab from {}'.format((vocab_json_path)))
        vocab = load_vocab(vocab_json_path)

        data_src = kwargs.pop('data_src', None)

        self.question_filename = kwargs.pop('question_file')
        self.question_id_info_filename = self.question_filename + '.ids.json'

        q_image_ids, q_idxs, _ = load_id_data(self.question_id_info_filename, data_src)

        self.feature_h5_path = kwargs.pop('feature_h5')

        use_spatial = kwargs.pop('spatial')

        with h5py.File(self.feature_h5_path, 'r') as features_file:
            coco_ids = features_file['ids'][()]
        feat_coco_id_to_index = {id: i for i, id in enumerate(coco_ids)}

        self.dataset = VQADataset(
            self.question_filename,
            feat_coco_id_to_index,
            q_image_ids,
            q_idxs,
            self.feature_h5_path,
            len(vocab['answer_token_to_idx']),
            use_spatial,
            get_boxes=kwargs.pop('get_boxes', False)
        )

        self.vocab = vocab
        self.batch_size = kwargs['batch_size']

        kwargs['collate_fn'] = default_collate
        super().__init__(self.dataset, **kwargs)

    def __len__(self):
        return math.ceil(len(self.dataset) / self.batch_size)


class PerturbVQADataLoader(DataLoader):
    def __init__(self, **kwargs):
        vocab_json_path = str(kwargs.pop('vocab_json', ''))
        if vocab_json_path:
            print('loading vocab from %s' % (vocab_json_path))
            vocab = load_vocab(vocab_json_path)
        else:
            vocab = None

        perturb_type = kwargs.pop('perturb_type')

        self.question_filename = kwargs.pop('question_file')
        self.question_id_info_filename = self.question_filename + '.ids.json'

        q_image_ids, q_idxs, og_q_idxs = load_id_data(self.question_id_info_filename, perturb_type)

        self.feature_h5_path = kwargs.pop('feature_h5')

        use_spatial = kwargs.pop('spatial')

        with h5py.File(self.feature_h5_path, 'r') as features_file:
            coco_ids = features_file['ids'][()]
        feat_coco_id_to_index = {id: i for i, id in enumerate(coco_ids)}

        self.dataset = PerturbVQADataset(
            self.question_filename,
            feat_coco_id_to_index,
            q_image_ids,
            q_idxs,
            og_q_idxs,
            self.feature_h5_path,
            kwargs.pop('num_answers') if not vocab else len(vocab['answer_token_to_idx']),
            use_spatial,
            get_boxes=False
        )

        self.vocab = vocab
        self.batch_size = kwargs['batch_size']

        kwargs['collate_fn'] = default_collate
        super().__init__(self.dataset, **kwargs)

    def __len__(self):
        return math.ceil(len(self.dataset) / self.batch_size)


class IdentityIndexer(object):
    def __init__(self):
        pass

    def __getitem__(self, item):
        return item
