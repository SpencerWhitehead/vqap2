import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # to import shared utils
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import argparse, yaml
import random
import logging
from collections import Counter

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s')
logFormatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
rootLogger = logging.getLogger()

from multiDataLoader import load_glove_embeddings, init_dataloader, init_perturb_dataloaders
from model.xnm_net import XNMNet
from utils.misc import todevice
from validate import validate


def seed_all(rseed):
    torch.manual_seed(rseed)
    torch.cuda.manual_seed(rseed)
    np.random.seed(rseed)
    random.seed(rseed)


def sample_task(counts):
    if not counts:
        return None

    if len(counts) == 1:
        return list(counts.keys())[0]

    task_list = list(filter(lambda k: counts[k] > 0, counts))
    if task_list:
        return random.choice(task_list)
    else:
        return None


def reset_task_counts(data_loaders):
    if not data_loaders:
        return Counter()
    return Counter({k: len(d_l) for k, d_l in data_loaders.items()})


def sample_perturb_batch(counts, data_loaders, data_loader_iters, model, device, l_module, epsilon=1e-8, loss_fn=None):
    if l_module <= 0.0 or not counts or not data_loaders or not data_loader_iters:
        return counts, data_loaders, data_loader_iters, 0.0

    task_label = sample_task(counts)
    if task_label is None:
        counts = reset_task_counts(data_loaders)
        task_label = sample_task(counts)

    counts[task_label] -= 1

    try:
        perturb_batch = next(data_loader_iters[task_label])
    except StopIteration:
        data_loader_iters[task_label] = iter(
            data_loaders[task_label]
        )
        perturb_batch = next(data_loader_iters[task_label])

    _, *batch_input = [todevice(x, device) for x in perturb_batch]

    (
        pert_answers,
        pert_questions,
        pert_question_len,
        vision_feat,
        relation_mask,
        og_answers,
        og_questions,
        og_question_len
    ) = batch_input

    pert_logits, pert_others = model(pert_questions, pert_question_len, vision_feat, relation_mask)
    pert_module_prob = torch.stack(pert_others['module_prob'], dim=0).permute(2, 0, 1)

    og_logits, og_others = model(og_questions, og_question_len, vision_feat, relation_mask)
    og_module_prob = torch.stack(og_others['module_prob'], dim=0).permute(2, 0, 1)

    if loss_fn is None:
        log_pert_module_prob = torch.log(pert_module_prob + epsilon)
        module_select_loss = nn.functional.kl_div(log_pert_module_prob, og_module_prob, reduction='batchmean')
    else:
        module_select_loss = loss_fn(og_module_prob, pert_module_prob)

    pert_loss = l_module * module_select_loss

    return counts, data_loaders, data_loader_iters, pert_loss


def get_perturbed_data(args, num_answers):
    l_module = 0.0
    perturb_prob = 0.0
    pert_loss_func = None
    perturb_loaders = {}
    perturb_loader_iters = {}
    perturb_batch_cnt = Counter()

    arg_names = ['perturb_prob', 'perturbations', 'l_module', 'pert_loss_func']
    has_all_attrs = [hasattr(args, aname) for aname in arg_names]
    if all(has_all_attrs):
        if args.perturb_prob > 0.0 and args.perturbations:
            logging.info('Initializing perturbed data...')

            l_module = args.l_module

            perturb_train_loader_kwargs = {
                'question_file': args.train_question_file,
                'num_answers': num_answers,
                'feature_h5': args.feature_h5,
                'batch_size': args.batch_size,
                'spatial': args.spatial,
                'num_workers': 2,
                'shuffle': True,
                'perturbations': args.perturbations
            }
            perturb_loaders, _, _, perturb_loader_iters = \
                init_perturb_dataloaders(perturb_train_loader_kwargs, get_iters=True)

            perturb_batch_cnt = reset_task_counts(perturb_loaders)

            perturb_prob = args.perturb_prob
            pert_loss_func = None
            if args.pert_loss_func == 'l1':
                pert_loss_func = nn.L1Loss()
            elif args.pert_loss_func == 'l2':
                pert_loss_func = nn.MSELoss()

            logging.info('perturbed loader length: {}'.format(perturb_batch_cnt))
        else:
            logging.info(
                'Not initializing perturbed data.\n\tperturb_prob = {}\n\tperturbations = {}'.format(args.perturb_prob,
                                                                                                     args.perturbations)
            )
    else:
        logging.info('No perturbed data will be loaded.')
        for a_idx in range(len(arg_names)):
            if not has_all_attrs[a_idx]:
                logging.info('\tArgument "{}" not provided.'.format(arg_names[a_idx]))

    return (l_module,
            perturb_prob,
            pert_loss_func,
            perturb_loaders,
            perturb_loader_iters,
            perturb_batch_cnt)


def train(args):
    if not hasattr(args, 'temperature'):
        logging.info('By default, using temperature = 1 for gumbel noise')
        args.temperature = 1.0

    logging.info('Create data loaders.........')
    train_loader_kwargs = {
        'question_file': args.train_question_file,
        'vocab_json': args.vocab_json,
        'feature_h5': args.feature_h5,
        'batch_size': args.batch_size,
        'spatial': args.spatial,
        'num_workers': 2,
        'shuffle': True,
        'data_src': args.data_src
    }
    train_loader, _, _ = init_dataloader(args.dataset, train_loader_kwargs)

    (
        l_module,
        perturb_prob,
        pert_loss_func,
        perturb_loaders,
        perturb_loader_iters,
        perturb_batch_cnt
    ) = get_perturbed_data(args, num_answers=len(train_loader.vocab['answer_token_to_idx']))

    val_loader_kwargs = {
        'question_file': args.eval_question_file,
        'vocab_json': args.vocab_json,
        'feature_h5': args.feature_h5,
        'batch_size': args.batch_size,
        'spatial': args.spatial,
        'num_workers': 2,
        'shuffle': False,
        'data_src': ['original']
    }
    val_loader, _, _ = init_dataloader(args.dataset, val_loader_kwargs)

    logging.info('train loader length: {}'.format(len(train_loader)))
    logging.info('eval loader length: {}'.format(len(val_loader)))
    logging.info('train data length: {}'.format(len(train_loader.dataset)))
    logging.info('eval data length: {}'.format(len(val_loader.dataset)))

    logging.info('Create model.........')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_kwargs = {
        'vocab': train_loader.vocab,
        'dim_v': args.dim_v,
        'dim_mem': args.dim_mem,
        'dim_word': args.dim_word,
        'dim_hidden': args.dim_hidden,
        'dim_vision': args.dim_vision,
        'dim_edge': args.dim_edge,
        'cls_fc_dim': args.cls_fc_dim,
        'dropout_prob': args.dropout,
        'T_ctrl': args.T_ctrl,
        'glimpses': args.glimpses,
        'stack_len': args.stack_len,
        'device': device,
        'spatial': args.spatial,
        'use_gumbel': args.module_prob_use_gumbel == 1,
        'temperature': args.temperature,
        'use_validity': args.module_prob_use_validity == 1,
        'is_norm_fc_module': args.is_norm_fc_module
    }
    model_kwargs_tosave = {k: v for k, v in model_kwargs.items() if k != 'vocab'}

    model = XNMNet(**model_kwargs).to(device)

    logging.info(model)

    logging.info('load glove vectors')
    glove_filename = args.train_question_file + '.glove.p'
    model = load_glove_embeddings(glove_filename, model, device)

    ################################################################

    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(parameters, args.lr, weight_decay=0)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.5 ** (1 / args.lr_halflife))

    start_epoch = 0
    t_since_improvement = 0
    best_acc = 0.

    if args.resume:
        logging.info('Restore checkpoint and optimizer...')
        ckpt_path = os.path.join(args.save_dir, 'best_model.pt')
        ckpt = torch.load(ckpt_path, map_location={'cuda:0': 'cpu'})
        start_epoch = ckpt['epoch'] + 1
        model.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        best_acc = ckpt['best_score']

        logging.info('Best score from previous run: {}'.format(best_acc))
        logging.info('Starting from epoch {}'.format(start_epoch))

    seed_all(args.seed)

    logging.info('Start training........')
    for epoch in range(start_epoch, args.num_epoch):
        model.train()
        valid_acc = 0.

        for i, batch in enumerate(train_loader):
            progress = epoch + i / len(train_loader)
            coco_ids, answers, *batch_input = [todevice(x, device) for x in batch]
            logits, others = model(*batch_input)
            ##################### Answer loss #####################
            nll = -nn.functional.log_softmax(logits, dim=1)
            loss = (nll * answers / 10).sum(dim=1).mean()

            if perturb_loaders and random.random() <= perturb_prob and l_module > 0.0:
                ##################### Q3R loss #####################
                (
                    perturb_batch_cnt,
                    perturb_loaders,
                    perturb_loaders_iters,
                    pert_loss
                ) = sample_perturb_batch(
                    perturb_batch_cnt,
                    perturb_loaders,
                    perturb_loader_iters,
                    model,
                    device,
                    l_module,
                    loss_fn=pert_loss_func
                )
                loss += pert_loss

            #################################################
            scheduler.step()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_value_(parameters, clip_value=0.5)
            optimizer.step()
            if (i + 1) % (len(train_loader) // args.print_per_epoch) == 0:
                curr_lr = []
                for param_group in optimizer.param_groups:
                    curr_lr.append(param_group['lr'])
                logging.info('Progress {:.3f}  ce_loss = {:.3f}  lr = {:.5f}'.format(progress, loss.item(), curr_lr[0]))

            if (i + 1) % (len(train_loader) // args.validation_per_epoch) == 0:
                valid_acc = validate(model, val_loader, device)
                logging.info('\n ~~~~~~ Progress %.3f  Valid Accuracy: %.4f ~~~~~~~\n' % (progress, valid_acc))
                model.train()

                if valid_acc > best_acc:
                    best_acc = valid_acc
                    t_since_improvement = 0

                    save_checkpoint(
                        epoch,
                        model,
                        optimizer,
                        scheduler,
                        model_kwargs_tosave,
                        os.path.join(args.save_dir, 'best_model_{}.pt'.format(epoch)),
                        valid_acc,
                        best_acc,
                        val_loader_kwargs['data_src']
                    )
                    logging.info(' >>>>>> save best to {} <<<<<<'.format(args.save_dir))

                else:
                    t_since_improvement += 1
                    logging.info(' >>>>>> evaluations since last improvement: {} <<<<<<'.format(t_since_improvement))

        save_checkpoint(
            epoch,
            model,
            optimizer,
            scheduler,
            model_kwargs_tosave,
            os.path.join(args.save_dir, 'model.pt'),
            valid_acc,
            best_acc,
            val_loader_kwargs['data_src']
        )
        logging.info(' >>>>>> save to {} <<<<<<'.format(args.save_dir))


def save_checkpoint(epoch, model, optimizer, scheduler, model_kwargs, filename, current_score, best_score, validation_data):
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'model_kwargs': model_kwargs,
        'current_score': current_score,
        'best_score': best_score,
        'eval_data': validation_data
    }
    torch.save(state, filename)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--save_dir', default='output/checkpoints/')
    parser.add_argument('--input_dir', default='output/preprocessed/')
    parser.add_argument('--train_question_file', default='train_questions.h5')
    parser.add_argument('--eval_question_file', default='eval_questions.h5')
    parser.add_argument('--vocab_json', default='vocab.json')
    parser.add_argument('--feature_h5', default='traineval_feature.h5')
    parser.add_argument('--dataset', default='VQA2')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--print_per_epoch', default=50, type=int)
    parser.add_argument('--validation_per_epoch', default=1, type=int)

    # model hyperparameters
    parser.add_argument('--dim_word', default=300, type=int, help='word embedding')
    parser.add_argument('--dim_v', default=512, type=int, help='node embedding')
    parser.add_argument('--dim_edge', default=256, type=int, help='edge embedding')
    parser.add_argument('--dim_vision', default=2048, type=int)
    parser.add_argument('--glimpses', default=2, type=int)
    parser.add_argument('--cls_fc_dim', default=1024, type=int, help='classifier fc dim')
    parser.add_argument('--dropout', default=0.5, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--spatial', action='store_true')

    parser.add_argument('--config', default='experiments/configs/config_XNM_copied_preproc.yaml')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    for k, v in config.items():
        setattr(args, k, v)

    args.train_question_file = os.path.join(args.input_dir, args.train_question_file)
    args.vocab_json = os.path.join(args.input_dir, args.vocab_json)
    args.eval_question_file = os.path.join(args.input_dir, args.eval_question_file)

    # make logging.info display into both shell and file
    if not args.resume:
        if not os.path.isdir(args.save_dir):
            os.mkdir(args.save_dir)
    else:
        assert os.path.isdir(args.save_dir)

    fileHandler = logging.FileHandler(os.path.join(args.save_dir, 'stdout.log'))
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)

    # args display
    logging.info('----------')
    logging.info('Arguments:')
    argvar_list = sorted([arg for arg in vars(args)])
    for arg in argvar_list:
        logging.info('\t{}: {}'.format(arg, getattr(args, arg)))
    logging.info('----------\n')

    # set random seed
    seed_all(args.seed)

    if args.spatial:
        args.dim_vision += 5

    train(args)


if __name__ == '__main__':
    main()
