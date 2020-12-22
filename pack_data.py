import os
import argparse
from utils.misc import load_json, save_json


def main(args):
    assert not os.path.exists(args.out_ques_file), \
        'Output QUESTION file already exists: {}'.format(args.out_ques_file)

    assert not os.path.exists(args.out_ann_file), \
        'Output ANNOTATION file already exists: {}'.format(args.out_ann_file)

    og_questions = load_json(args.og_ques_file)['questions']
    og_annotations = load_json(args.og_ann_file)['annotations']

    og_qid2idx = {}

    for i, (og_ques, og_ann) in enumerate(zip(og_questions, og_annotations)):
        assert og_ques['question_id'] == og_ann['question_id']
        assert og_ques['image_id'] == og_ann['image_id']

        og_ques['src'] = 'original'
        og_ques['perturbation'] = 'original'
        og_ques['original_id'] = og_ques['question_id']
        og_ques['original_index'] = i

        og_ann['src'] = 'original'
        og_ann['perturbation'] = 'original'
        og_ann['original_id'] = og_ann['question_id']
        og_ann['original_index'] = i

        og_qid2idx[og_ques['question_id']] = i

    if args.pert_ques_file and args.pert_ann_file:
        pert_questions = load_json(args.pert_ques_file)['questions']
        pert_annotations = load_json(args.pert_ann_file)['annotations']
        for pert_ques, pert_ann in zip(pert_questions, pert_annotations):
            assert pert_ques['question_id'] == pert_ann['question_id']
            assert pert_ques['image_id'] == pert_ann['image_id']

            pert_ques['original_index'] = og_qid2idx[pert_ques['original_id']]
            pert_ann['original_index'] = og_qid2idx[pert_ann['original_id']]
    else:
        pert_questions = []
        pert_annotations = []

    all_questions = og_questions + pert_questions
    save_json(args.out_ques_file, {'questions': all_questions})

    all_annotations = og_annotations + pert_annotations
    save_json(args.out_ann_file, {'annotations': all_annotations})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--og_ques_file', default='data/v2_OpenEnded_mscoco_val2014_questions.json')
    parser.add_argument('--og_ann_file', default='data/v2_mscoco_val2014_annotations.json')
    parser.add_argument('--pert_ques_file', default='data/vqap2.questions.json')
    parser.add_argument('--pert_ann_file', default='data/vqap2.annotations.json')
    parser.add_argument('--out_ques_file', default='data/eval_combined.questions.json')
    parser.add_argument('--out_ann_file', default='data/eval_combined.annotations.json')

    args = parser.parse_args()

    print('----------')
    print('Arguments:')
    argvar_list = sorted([arg for arg in vars(args)])
    for arg in argvar_list:
        print('\t{}: {}'.format(arg, getattr(args, arg)))
    print('----------\n')

    main(args)
