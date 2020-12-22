import os,sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from tqdm import tqdm
import argparse
import os
import json
import pickle
import itertools
from collections import OrderedDict
from utils.misc import todevice


def batch_accuracy(predicted, true):
    """ Compute the accuracies for a batch of predictions and answers """
    _, predicted_index = predicted.max(dim=1, keepdim=True)
    agreeing = true.gather(dim=1, index=predicted_index)
    '''
    Acc needs to be averaged over all 10 choose 9 subsets of human answers.
    While we could just use a loop, surely this can be done more efficiently (and indeed, it can).
    There are two cases for the 1 chosen answer to be discarded:
    (1) the discarded answer is not the predicted answer => acc stays the same
    (2) the discarded answer is the predicted answer => we have to subtract 1 from the number of agreeing answers
    
    There are (10 - num_agreeing_answers) of case 1 and num_agreeing_answers of case 2, thus
    acc = ((10 - agreeing) * min( agreeing      / 3, 1)
           +     agreeing  * min((agreeing - 1) / 3, 1)) / 10
    
    Let's do some more simplification:
    if num_agreeing_answers == 0:
        acc = 0  since the case 1 min term becomes 0 and case 2 weighting term is 0
    if num_agreeing_answers >= 4:
        acc = 1  since the min term in both cases is always 1
    The only cases left are for 1, 2, and 3 agreeing answers.
    In all of those cases, (agreeing - 1) / 3  <  agreeing / 3  <=  1, so we can get rid of all the mins.
    By moving num_agreeing_answers from both cases outside the sum we get:
        acc = agreeing * ((10 - agreeing) + (agreeing - 1)) / 3 / 10
    which we can simplify to:
        acc = agreeing * 0.3
    Finally, we can combine all cases together with:
        min(agreeing * 0.3, 1)
    '''
    return (agreeing * 0.3).clamp(max=1)


def validate(model, data, device, evalonly=False):
    model.eval()
    print('validate...')
    results_acc = []
    total_acc, count = 0, 0
    with torch.no_grad():
        for batch in tqdm(data, total=len(data)):
            coco_ids, answers, *batch_input = [todevice(x, device) for x in batch]
            logits, others = model(*batch_input)
            acc = batch_accuracy(logits, answers)

            if evalonly: # for evaluation only mode, don't need to save this during training
                for a in acc:
                    results_acc.append(a.item())

            total_acc += acc.sum().item()
            count += answers.size(0)

        acc = total_acc / count
    if not evalonly:
        return acc
    else:
        return acc, results_acc


def test(model, data, device):
    model.eval()
    results = []
    with torch.no_grad():
        for batch in tqdm(data, total=len(data)):
            coco_ids, answers, *batch_input = [todevice(x, device) for x in batch]
            logits, others = model(*batch_input)
            predicts = torch.max(logits, dim=1)[1]
            for predict in predicts:
                results.append(data.vocab['answer_idx_to_token'][predict.item()])
    return results


def evaluate_subsets(result_accs, subset_id_data):
    collective_acc = sum(result_accs) / len(result_accs)

    allperturbtype_pert_q_idxs = subset_id_data['perturb_q_idxs']
    allperturbtype_og_q_idxs = subset_id_data['original_q_idxs']

    subset_accs = OrderedDict()
    subset_cons = OrderedDict()

    subset_accs['all'] = {'original': collective_acc, 'perturb': collective_acc}

    p2pairs_og_acc = 0.  # total accuracy on original questions that are part of perturbed pairs
    p2pairs_pert_acc = 0.  # total accuracy on perturbed questions that are part of perturbed pairs
    num_pert_pairs = 0

    all_groupings_list = []

    subset_names = sorted(list(allperturbtype_pert_q_idxs.keys()))

    for subset_name in subset_names:
        subset_pertqidxs = allperturbtype_pert_q_idxs.get(subset_name, [])
        subset_ogqidxs = allperturbtype_og_q_idxs.get(subset_name, [])

        assert len(subset_pertqidxs) == len(subset_ogqidxs)

        if subset_pertqidxs and subset_ogqidxs:
            subset_pert_acc = 0.
            subset_og_acc = 0.
            subset_total_acc = 0.

            subset_groupings_list = []
            for og_idx, pert_idx in zip(subset_ogqidxs, subset_pertqidxs):
                og_q_acc = result_accs[og_idx]
                pert_q_acc = result_accs[pert_idx]
                subset_og_acc += og_q_acc
                subset_pert_acc += pert_q_acc
                subset_total_acc += og_q_acc + pert_q_acc

                if subset_name not in {'original'}:
                    subset_groupings_list.append((og_idx, [pert_idx]))
                    all_groupings_list.append((og_idx, [pert_idx]))

            if subset_name not in {'original'}:
                p2pairs_og_acc += subset_og_acc
                p2pairs_pert_acc += subset_pert_acc
                num_pert_pairs += len(subset_ogqidxs)

            subset_og_acc /= len(subset_ogqidxs)
            subset_pert_acc /= len(subset_pertqidxs)
            subset_total_acc /= (len(subset_ogqidxs) + len(subset_pertqidxs))
            mean_total_acc = (subset_og_acc + subset_pert_acc) / 2.

            subset_accs[subset_name] = {
                'original': subset_og_acc,
                'perturb': subset_pert_acc,
                'total': subset_total_acc,
                'mean': mean_total_acc
            }

            if subset_groupings_list:
                current_cons, _, _, _, _, _ = evaluate_consistency(result_accs, subset_groupings_list, k_vals=(1, 2))
                subset_cons[subset_name] = current_cons

    p2pairs_og_acc /= num_pert_pairs
    p2pairs_pert_acc /= num_pert_pairs

    if all_groupings_list:
        current_cons, _, _, _, _, _ = evaluate_consistency(result_accs, all_groupings_list, k_vals=(1, 2))
        subset_cons['p2pairs_perturb'] = current_cons

    subset_accs['p2pairs_original'] = {'original': p2pairs_og_acc, 'perturb': p2pairs_og_acc}
    subset_accs['p2pairs_perturb'] = {'original': p2pairs_pert_acc, 'perturb': p2pairs_pert_acc}

    return subset_accs, subset_cons


def calc_consistency(Q_accs, k_val):
    """
    Implementation of Consensus Score (CS) formula from Shah et al. (2019): https://arxiv.org/abs/1902.05660
    :param Q_accs: List of accuracies for a question grouping
    :param k_val: k from CS definition
    :return:
    """
    total_Q_S = 0.
    n_C_k = 0.
    for Q_prime in itertools.combinations(Q_accs, k_val):
        n_C_k += 1.
        total_Q_S += float(all(score > 0. for score in Q_prime))
    return total_Q_S / n_C_k


def evaluate_consistency(result_accs, qgroupings, k_vals=(1, 2, 3, 4)):
    """
    Compute Consensus Scores (CS) from Shah et al. (2019): https://arxiv.org/abs/1902.05660
    :param result_accs: A list of VQA accuracies for all questions
    :param qgroupings: A dictionary or list of tuples of {original_index: [perturbed_index1, perturbed_index2, ...], }
    :param k_vals: k from the CS definition
    """
    collective_acc = sum(result_accs) / len(result_accs)

    allk_total_cons = {}
    allk_results_cons = {}
    num_groupings = len(qgroupings)
    num_og_questions = 0
    num_pert_questions = 0

    og_acc = 0.
    pert_acc = 0.

    if isinstance(qgroupings, dict):
        groupings_list = list(qgroupings.items())
    else:
        groupings_list = qgroupings

    for i, csk in enumerate(k_vals):
        total_cons = 0.
        results_cons = []

        for og_idx, pert_indices in groupings_list:
            group_og_acc = [result_accs[og_idx]]
            group_pert_acc = [result_accs[gqidx] for gqidx in pert_indices]
            group_accs = group_og_acc + group_pert_acc
            group_cons = calc_consistency(group_accs, csk)
            total_cons += group_cons
            results_cons.append(group_cons)

            if i == 0:
                og_acc += sum(group_og_acc)
                num_og_questions += len(group_og_acc)

                pert_acc += sum(group_pert_acc)
                num_pert_questions += len(group_pert_acc)

        allk_total_cons[csk] = total_cons / num_groupings
        allk_results_cons[csk] = results_cons

    og_acc /= num_og_questions
    pert_acc /= num_pert_questions

    mean_pert_acc = (og_acc + pert_acc) / 2.

    return allk_total_cons, allk_results_cons, og_acc, pert_acc, collective_acc, mean_pert_acc


def display_program_args(args, outfile):
    outfile.write('----------\n')
    outfile.write('Arguments:\n')
    for arg in vars(args):
        outfile.write('\t{}: {}\n'.format(arg, getattr(args, arg)))
    outfile.write('----------\n\n')


def display_accuracies(acc_dict, args, outfile, show_args=True, consistencies=()):
    if show_args:
        display_program_args(args, outfile)

    outfile.write('\n')
    for perturb_type, perturb_type_accs in acc_dict.items():
        og_acc, p_acc = perturb_type_accs['original'], perturb_type_accs['perturb']
        total_acc, mean_acc = perturb_type_accs.get('total', None), perturb_type_accs.get('mean', None)
        outfile.write(
            '{}\n\tOriginal accuracy: {:.4f}\n\tPerturbed accuracy: {:.4f}\n'.format(perturb_type, og_acc, p_acc)
        )
        if mean_acc is not None:
            outfile.write(
                '\n\tMean accuracy: {:.4f}\n'.format(mean_acc)
            )
        if total_acc is not None:
            outfile.write(
                '\n\tCollective accuracy: {:.4f}\n'.format(total_acc)
            )

        if perturb_type in consistencies:
            display_consistencies(
                consistencies[perturb_type], None, None, None, None, None, outfile, show_args=False
            )
        else:
            outfile.write(('=' * 50) + '\n\n')


def display_consistencies(cons_dict, original_acc, pert_acc, collective_acc, mean_pert_acc, args, outfile, **kwargs):
    if kwargs.get('show_args', True):
        display_program_args(args, outfile)

    outfile.write('\n')
    if collective_acc is not None:
        outfile.write('Collective accuracy: {:.4f}\n'.format(collective_acc))

    if pert_acc is not None:
        outfile.write('Mean accuracy ((Ori + Rep) / 2): {:.4f}\n'.format(mean_pert_acc))

    outfile.write('\n')
    for K, cs_score in cons_dict.items():
        outfile.write(
            '\tCS(k={}): {:.4f}\n'.format(K, cs_score)
        )

    min_K, max_K = min(cons_dict.keys()), max(cons_dict.keys())
    outfile.write('\n')
    outfile.write('\tCS(k={}) - CS(k={}): {:.4f}\n'.format(min_K, max_K, cons_dict[min_K] - cons_dict[max_K]))

    if original_acc is not None and pert_acc is not None:
        outfile.write(
            '\n\tOriginal accuracy: {:.4f}\n\tPerturbed accuracy: {:.4f}\n'.format(original_acc, pert_acc)
        )
        outfile.write('\tOriginal - Perturbed: {:.4f}\n'.format(original_acc - pert_acc))
    outfile.write(('=' * 50) + '\n\n')


def run_accuracy_evals(result_accs, args):
    with open(args.question_ids, 'r') as iddataf:
        eval_id_data = json.load(iddataf)

    accuracies, consistency_scores = evaluate_subsets(result_accs, eval_id_data)

    display_accuracies(accuracies, args, sys.stdout, consistencies=consistency_scores)

    if args.output_file:
        with open(args.output_file, 'w') as outf:
            display_accuracies(accuracies, args, outf, consistencies=consistency_scores)

    if args.out_result_accs_file:
        with open(args.out_result_accs_file, 'wb') as ra_outf:
            pickle.dump(result_accs, ra_outf)


if __name__ == '__main__':
    from model.xnm_net import XNMNet
    from multiDataLoader import init_dataloader

    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True,
                        help='Either a pretrained model file or a file with accuracies for each question.')
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--question_file', default='eval_questions.h5')
    parser.add_argument('--question_ids', default='eval_questions.h5.ids.json')
    parser.add_argument('--vocab_json', default='vocab.json')
    parser.add_argument('--feature_h5', default='data/traineval_feature.h5')
    parser.add_argument('--output_file', required=True, help='Used for score logging.')
    parser.add_argument('--out_result_accs_file',
                        help='Name of file to output accuracies for each predicted answer.', default=None)
    args = parser.parse_args()

    args.vocab_json = os.path.join(args.input_dir, args.vocab_json)
    args.question_file = os.path.join(args.input_dir, args.question_file)
    args.question_ids = os.path.join(args.input_dir, args.question_ids)

    display_program_args(args, sys.stdout)

    device = 'cuda'
    loaded = torch.load(args.ckpt, map_location={'cuda:0': 'cpu'})
    model_kwargs = loaded['model_kwargs']

    loader_kwargs = {
        'batch_size': 256,
        'spatial': model_kwargs['spatial'],
        'num_workers': 2,
        'shuffle': False,
        'vocab_json': args.vocab_json,
        'question_file': args.question_file,
        'feature_h5': args.feature_h5,
        'data_src': None
    }

    data_loader, _, _ = init_dataloader('VQA2', loader_kwargs)

    model_kwargs.update({'vocab': data_loader.vocab, 'device': device})

    model = XNMNet(**model_kwargs).to(device)
    model.load_state_dict(loaded['state_dict'])

    collective_acc, result_accs = validate(model, data_loader, device, evalonly=True)

    run_accuracy_evals(result_accs, args)
