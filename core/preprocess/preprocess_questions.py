import re
import os
import argparse
import json
import numpy as np
import pickle
from utils import encode
from collections import Counter, defaultdict
import h5py


# Please see: [https://github.com/Cyanogenoid/vqa-counting/blob/master/vqa-v2/data.py]
_special_chars = re.compile('[^a-z0-9 ]*')
_period_strip = re.compile(r'(?!<=\d)(\.)(?!\d)')
_comma_strip = re.compile(r'(\d)(,)(\d)')
_punctuation_chars = re.escape(r';/[]"{}()=+\_-><@`,?!')
_punctuation = re.compile(r'([{}])'.format(re.escape(_punctuation_chars)))
_punctuation_with_a_space = re.compile(r'(?<= )([{0}])|([{0}])(?= )'.format(_punctuation_chars))


def process_punctuation(s):
    if _punctuation.search(s) is None:
        return s
    s = _punctuation_with_a_space.sub('', s)
    if re.search(_comma_strip, s) is not None:
        s = s.replace(',', '')
    s = _punctuation.sub(' ', s)
    s = _period_strip.sub('', s)
    return s.strip()


def process_answers(annotation_list, topn_answers=None, collect_vocab=False):
    answer_cnt = Counter()
    for ann in annotation_list:
        answers = [ansr['answer'] for ansr in ann['answers']]

        for i, answer in enumerate(answers):
            answer = process_punctuation(answer)
            answers[i] = answer
            if collect_vocab:
                answer_cnt[answer] += 1

        ann['answers'] = answers  # update

    if collect_vocab:
        answer_token_to_idx = {}

        for token, cnt in answer_cnt.most_common(topn_answers):
            answer_token_to_idx[token] = len(answer_token_to_idx)

        return annotation_list, answer_token_to_idx
    else:
        return annotation_list


def gather_input_data(input_question_json, input_annotation_json, output_fname):
    questions, annotations = [], []
    perturb_image_ids = defaultdict(list)
    perturb_question_idxs = defaultdict(list)
    perturb_og_question_idxs = defaultdict(list)

    with open(input_question_json, 'r') as f:
        question_data = json.load(f)['questions']

    if input_annotation_json:
        for fname in input_annotation_json.split(':'):
            with open(fname, 'r') as f:
                annotations += json.load(f)['annotations']

        assert len(question_data) == len(annotations), \
            '{} questions\t{} annotations'.format(len(question_data), len(annotations))

    image_ids = []
    ids = []

    for i, q in enumerate(question_data):
        image_ids.append(q['image_id'])
        ids.append(q['question_id'])
        questions.append(q['question'])

        perturb_type = q.get('perturbation', 'original')
        img_id = q['image_id']
        orig_index = q.get('original_index', i)

        perturb_image_ids[perturb_type].append(img_id)
        perturb_question_idxs[perturb_type].append(i)
        perturb_og_question_idxs[perturb_type].append(orig_index)

    id_map_name = output_fname + '.ids.json'
    id_data = {
        'question_ids': ids,
        'image_ids': image_ids,
        'perturb_image_ids': dict(perturb_image_ids),
        'perturb_q_idxs': dict(perturb_question_idxs),
        'original_q_idxs': dict(perturb_og_question_idxs)
    }
    with open(id_map_name, 'w', encoding='utf-8') as outf:
        json.dump(id_data, outf)

    return questions, annotations


def main(args):
    print('Loading data')
    questions, annotations = gather_input_data(
        args.input_questions_json,
        args.input_annotations_json,
        args.output_filename
    )

    if args.mode == 'train':
        annotations, answer_token_to_idx = process_answers(
            annotations, topn_answers=args.answer_top, collect_vocab=True
        )

        print('Get answer_token_to_idx, num: {}'.format(len(answer_token_to_idx)))

        question_token_to_idx = {'<NULL>': 0, '<UNK>': 1}

        for i,q in enumerate(questions):
            question = q.lower()
            question = _special_chars.sub('', question)
            questions[i] = question

            for token in question.split(' '):
                if token not in question_token_to_idx:
                    question_token_to_idx[token] = len(question_token_to_idx)

        print('Get question_token_to_idx (size = {})'.format(len(question_token_to_idx)))

        vocab = {
            'question_token_to_idx': question_token_to_idx,
            'answer_token_to_idx': answer_token_to_idx,
            'program_token_to_idx': {token: i for i, token in
                                     enumerate(['<eos>', 'find', 'relate', 'describe', 'is', 'and'])},
        }

        print('Write into {}'.format(args.vocab_json))
        with open(args.vocab_json, 'w') as f:
            json.dump(vocab, f, indent=4)
    else:
        print('Load existing vocab ...')
        with open(args.vocab_json, 'r') as vocf:
            vocab = json.load(vocf)

        for ann in annotations:
            answers = [ansr['answer'] for ansr in ann['answers']]

            for i, answer in enumerate(answers):
                answer = process_punctuation(answer)
                answers[i] = answer

            ann['answers'] = answers  # update

        for i, q in enumerate(questions):
            question = q.lower()
            question = _special_chars.sub('', question)
            questions[i] = question

    # Encode all questions
    print('Encoding data')
    questions_encoded = []
    questions_len = []
    answers = []

    if args.mode in {'train', 'eval'}:
        for i, a in enumerate(annotations):
            question = questions[i]

            question_tokens = question.split(' ')
            question_encoded = encode(question_tokens, vocab['question_token_to_idx'], allow_unk=True)
            assert len(question_tokens) == len(question_encoded)
            questions_encoded.append(question_encoded)
            questions_len.append(len(question_encoded))

            answer = []

            for per_ans in a['answers']:
                if per_ans in vocab['answer_token_to_idx']:
                    answer.append(vocab['answer_token_to_idx'][per_ans])
                else:
                    answer.append(-1)
            answers.append(answer)
    elif args.mode in {'testdev', 'teststd'}:
        for i, q in enumerate(questions):  # Ensure the original order matches the question_id order
            question = questions[i]
            question_tokens = question.split(' ')
            question_encoded = encode(question_tokens, vocab['question_token_to_idx'], allow_unk=True)
            questions_encoded.append(question_encoded)
            questions_len.append(len(question_encoded))

            answers.append([-1])  # add dummy token for test

    # Pad encoded questions
    max_question_length = max(len(x) for x in questions_encoded)
    for qe in questions_encoded:
        while len(qe) < max_question_length:
            qe.append(vocab['question_token_to_idx']['<NULL>'])

    questions_encoded = np.asarray(questions_encoded, dtype=np.int32)
    questions_len = np.asarray(questions_len, dtype=np.int32)
    answers = np.array(answers, dtype=np.int32)
    print(
        'questions_encoded.shape: {}\t'.format(questions_encoded.shape),
        'questions_len.shape: {}\t'.format(questions_len.shape),
        'answers.shape: {}\t'.format(answers.shape)
    )

    print('Writing')

    # Save glove vectors
    if args.mode == 'train':
        glove_file_name = args.output_filename + '.glove.p'

        token_itow = {i: w for w, i in vocab['question_token_to_idx'].items()}
        print('Load glove from {}'.format(args.glove_pt))
        with open(args.glove_pt, 'rb') as glovef:
            glove = pickle.load(glovef)

        dim_word = glove['the'].shape[0]
        glove_matrix = []
        for i in range(len(token_itow)):
            glove_matrix.append(glove.get(token_itow[i], np.zeros((dim_word,))))
        glove_matrix = np.asarray(glove_matrix, dtype=np.float32)
        print(glove_matrix.shape)

        with open(glove_file_name, 'wb') as f:
            pickle.dump(glove_matrix, f)

    output_name = args.output_filename
    num_questions = len(questions)

    if os.path.isfile(output_name):
        os.remove(output_name)

    with h5py.File(output_name, 'w', libver='latest') as fd:
        h5_questions = fd.create_dataset('questions', data=questions_encoded)
        h5_answers = fd.create_dataset('answers', data=answers)

        h5_questions_len = fd.create_dataset('questions_len', shape=(num_questions, 1), dtype='int32')
        for i in range(num_questions):
            h5_questions_len[i] = int(questions_len[i])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--answer_top', default=3000, type=int)
    parser.add_argument('--glove_pt', default='data/glove/glove.840B.300d.pt')
    parser.add_argument('--input_questions_json', required=True,
                        help='data/v2_OpenEnded_mscoco_train2014_questions.json')
    parser.add_argument('--input_annotations_json',
                        help='data/v2_mscoco_train2014_annotations.json')
    parser.add_argument('--output_filename', required=True,
                        help='output/preprocessed/train_questions.h5')
    parser.add_argument('--vocab_json', required=True,
                        help='output/preprocessed/vocab.json')
    parser.add_argument('--mode', choices=['train', 'eval', 'testdev', 'teststd'], required=True)

    args = parser.parse_args()

    print('Running', os.path.basename(__file__))
    print('----------')
    print('Arguments:')
    for arg in vars(args):
        print('{}: {}'.format(arg, getattr(args, arg)))
    print('----------\n')

    main(args)
