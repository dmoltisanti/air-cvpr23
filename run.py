import argparse
import pickle
import sys
from pathlib import Path

import pandas as pd
import torch
from model_action_adverb_modifiers import act_adv_factory
from evaluator import Evaluator
from model import RegClsAdverbModel
from s3dg import S3D
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import Dataset, collate_variable_length_seq


def create_parser():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('train_df_path', type=Path)
    parser.add_argument('test_df_path', type=Path)
    parser.add_argument('antonyms_df', type=Path)
    parser.add_argument('dataset_pickle_path', type=Path)
    parser.add_argument('data_path', type=Path)
    parser.add_argument('output_path', type=Path)
    parser.add_argument('--checkpoint_path', default=None, type=Path)
    parser.add_argument('--train_batch', default=64, type=int)
    parser.add_argument('--train_workers', default=8, type=int)
    parser.add_argument('--test_batch', default=256, type=int)
    parser.add_argument('--test_workers', default=8, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--dropout', default=0.5, type=float)
    parser.add_argument('--weight_decay', default=5e-5, type=float)
    parser.add_argument('--epochs', default=1000, type=int)
    parser.add_argument('--run_tags', default=['lr', 'train_batch', 'dropout'], action='append')
    parser.add_argument('--tag', default=None, type=str)
    parser.add_argument('--train_only', action='store_true')
    parser.add_argument('--test_only', action='store_true')
    parser.add_argument('--test_frequency', default=10, type=int)
    parser.add_argument('--hidden_units', default='512,512,512', type=str)
    parser.add_argument('--s3d_init_folder', type=Path, default=None)
    parser.add_argument('--s3d_video_f', default='s3d_features', type=str)

    parser.add_argument('--no_antonyms', action='store_true')
    parser.add_argument('--fixed_d', action='store_true')
    parser.add_argument('--cls_variant', action='store_true')

    return parser


def setup_data(args):
    train_df = pd.read_csv(args.train_df_path)
    test_df = pd.read_csv(args.test_df_path)
    antonyms_df = pd.read_csv(args.antonyms_df)

    with open(args.dataset_pickle_path, 'rb') as f:
        dataset_pickle = pickle.load(f)

    feature_dim = 512 if args.s3d_video_f == 'video_embedding_joint_space' else 1024
    collate_fn = collate_variable_length_seq

    features_train, _ = load_features_set(args, 'rgb', 'train')
    features_test, _ = load_features_set(args, 'rgb', 'test')

    train_dataset = Dataset(train_df, antonyms_df, features_train, dataset_pickle, feature_dim,
                            no_antonyms=args.no_antonyms)
    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True,
                              pin_memory=True, num_workers=args.train_workers, collate_fn=collate_fn)

    test_dataset = Dataset(test_df, antonyms_df, features_test, dataset_pickle, feature_dim,
                           no_antonyms=args.no_antonyms)

    test_loader = DataLoader(test_dataset, batch_size=args.test_batch, shuffle=False, pin_memory=True,
                             num_workers=args.test_workers, collate_fn=collate_fn)

    return train_loader, test_loader


def load_features_set(args, feature_modality, set_):
    feature_dict = {}
    data_path = args.data_path / feature_modality / f'{set_}.pth'
    print(f'Loading features from {data_path}')
    fd = torch.load(data_path, map_location=torch.device('cpu'))
    features = fd['features']
    feature_dim = None

    for k, v in features.items():
        if isinstance(v, torch.Tensor):
            features[k] = v

            if feature_dim is None:
                feature_dim = v.shape[1]
        else:
            assert isinstance(v, dict)
            features[k] = {kk: vv for kk, vv in v.items()}

            if feature_dim is None:
                feature_dim = v[args.s3d_video_f].shape[1]

    feature_dict['features'] = features
    feature_dict['metadata'] = fd['metadata']

    return feature_dict, feature_dim


def setup_model(train_dataset, args, cuda=True):
    model = RegClsAdverbModel(train_dataset, args)

    if args.checkpoint_path is not None:
        print(f'Loading checkpoint {args.checkpoint_path}')
        state_dict = torch.load(args.checkpoint_path)
        state_dict = state_dict['model_state']
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=True)
        missing_keys = [k for k in missing_keys if not k.startswith('s3d_model')]
        assert len(missing_keys) == 0, missing_keys

    if torch.cuda.is_available() and cuda:
        model = model.cuda()
        model = torch.nn.DataParallel(model)

    return model


def setup_optimiser(model, args):
    optim_params = model.parameters()
    return torch.optim.Adam(optim_params, lr=args.lr, weight_decay=args.weight_decay)


def setup_criterion(args, train_dataset):
    # the loss is calculated already by the model
    def loss_wrapper(output, labels):
        loss = output[0]

        if loss is None:
            return None

        assert isinstance(loss, torch.Tensor)
        loss_out = loss

        return loss_out

    return loss_wrapper


def to_cuda(data_tuple):
    if torch.cuda.is_available():
        cuda_tuple = []

        for x in data_tuple:
            if isinstance(x, torch.Tensor):
                xc = x.cuda()
            elif isinstance(x, dict):
                xc = {k: v.cuda() if isinstance(v, torch.Tensor) and ('no_cuda' not in k) else v for k, v in x.items()}
            elif isinstance(x, list):
                xc = [item.cuda() for item in x]
            else:
                raise RuntimeError(f'Review this for type {type(x)}')

            cuda_tuple.append(xc)

        return tuple(cuda_tuple)
    else:
        return data_tuple


def train(loader, model, optimiser, criterion, args, evaluator, epoch=None):
    model.train()
    avg_loss, output, metadata = run_through_loader(model, loader, criterion, optimiser, args, evaluator,
                                                    tag='Training', epoch=epoch)
    return avg_loss, output, metadata


def run_through_loader(model, loader, criterion, optimiser, args, evaluator, tag, optimise=True, epoch=None, **kwargs):
    bar = tqdm(desc=tag, file=sys.stdout, total=len(loader))
    loss_batches = []
    all_output = {}
    all_metadata = None
    return_df = False

    for x_tuple in loader:
        batch_output, loss, batch_labels, batch_metadata = process_batch(x_tuple, model, criterion, args, optimise,
                                                                         epoch=epoch, **kwargs)
        batch_output_dict = get_output_dict(batch_output, batch_labels, loader.dataset, evaluator)

        if optimise:
            loss.backward()
            optimiser.step()
            optimiser.zero_grad()

        if loss is not None:
            bar.set_description(f'{tag} loss: {loss.detach().item():0.4f}')
            loss_batches.append(loss.detach().item())

        bar.update()

        for k, o in batch_output_dict.items():
            if o is None:
                continue

            if k not in all_output:
                all_output[k] = []

            all_output[k].append(o.detach())

        if all_metadata is None:
            all_metadata = {k: [] for k in batch_metadata.keys()}

        for k in all_metadata.keys():
            bv = batch_metadata[k]

            if isinstance(bv, list):
                all_metadata[k].extend(bv)
            elif isinstance(bv, torch.Tensor):
                all_metadata[k].extend(bv.tolist())
            else:
                raise RuntimeError(f'Define how to combine metadata for type {type(bv)}')

    avg_loss = torch.Tensor(loss_batches).mean().item()

    if not return_df:
        all_output = {k: torch.cat([x.unsqueeze(0) if x.ndim == 1 else x for x in v], dim=0) if v else None
                      for k, v in all_output.items()}

    return avg_loss, all_output, all_metadata


def get_output_dict(output, labels, dataset, evaluator, stack_scores=False):
    with torch.no_grad():
        predictions = output[1]
        predictions, predictions_no_act_gt = predictions
        adverb_gt, verb_gt = labels['adverb'], labels['verb']

        if stack_scores:
            scores_no_act_gt = torch.stack([predictions_no_act_gt[(adv, act)] for adv, act in dataset.pairs], 1)
        else:
            scores_no_act_gt = predictions_no_act_gt

        scores_no_act_gt_ant = evaluator.get_antonym_scores(scores_no_act_gt, adverb_gt)

        res_tuple = evaluator.get_scores(predictions, verb_gt, adverb_gt, stack_scores=stack_scores)
        scores, action_gt_scores, adverb_gt_scores, antonym_action_gt_scores, act_mask, adv_mask = res_tuple

        batch_size = len(adverb_gt)
        n_adverbs = len(dataset.adverbs)

        # adapted from test.py in action modifiers' code

        y_true_adverb = torch.zeros((batch_size, n_adverbs))
        y_score = torch.zeros((batch_size, n_adverbs))
        y_score_antonym = torch.zeros((batch_size, n_adverbs))
        y_score_no_act_gt = torch.zeros((batch_size, n_adverbs))
        y_score_antonym_no_act_gt = torch.zeros((batch_size, n_adverbs))

        for idx in range(batch_size):
            y_true_adverb[idx] = torch.Tensor([1 if dataset.adverb2idx[adv] == adverb_gt[idx] else 0
                                               for adv in dataset.adverbs])

            y_score[idx] = torch.Tensor(
                [action_gt_scores[idx][dataset.pairs.index((adv, dataset.idx2action[verb_gt[idx].item()]))]
                 for adv in dataset.adverbs])

            y_score_antonym[idx] = torch.Tensor(
                [antonym_action_gt_scores[idx][dataset.pairs.index((adv, dataset.idx2action[verb_gt[idx].item()]))]
                    for adv in dataset.adverbs])

            for ia, a in enumerate(dataset.adverbs):
                mask = dataset.get_verb_with_adverb_mask(a)
                y_score_no_act_gt[idx, ia] = scores_no_act_gt[idx, mask].max()
                y_score_antonym_no_act_gt[idx, ia] = scores_no_act_gt_ant[idx, mask].max()

        out_dict = dict(y_true_adverb=y_true_adverb, y_score_antonym=y_score_antonym,
                        y_score=y_score, scores=scores, scores_no_act_gt=scores_no_act_gt,
                        y_score_no_act_gt=y_score_no_act_gt, y_score_antonym_no_act_gt=y_score_antonym_no_act_gt)

        return out_dict


def process_batch(x_tuple, model, criterion, args, training, epoch=None, **kwargs):
    x, labels, metadata = x_tuple
    x, labels = to_cuda((x, labels))
    output = get_model_output(x, labels, metadata, model, args, training, epoch=epoch, **kwargs)
    loss = criterion(output, labels)
    return output, loss, labels, metadata


def get_model_output(x, labels, metadata, model, args, training, epoch=None, **kwargs):
    features = x
    adverbs = labels['adverb']
    actions = labels['verb']
    neg_adverbs = torch.LongTensor(metadata['negative_adverb']).cuda()
    sentences = metadata.get('sentence', None)
    adverb_pre_mapping = metadata.get('adverb_pre_mapping', None)
    ids = metadata['seg_id']
    tuple_in = (adverbs, actions, neg_adverbs, neg_actions, sentences, adverb_pre_mapping, ids)  # TODO update model input, remove sentences, neg_actions etc
    output = model(features, tuple_in, training=training, freeze_adv=freeze_adv)
    return output


def test(model, loader, criterion, args, evaluator):
    model.eval()

    with torch.no_grad():
        avg_loss, output, metadata = run_through_loader(model, loader, criterion, None, args, evaluator,
                                                        tag='Testing', optimise=False)

    return avg_loss, output, metadata


def compute_evaluation_metrics(output, args, dataset_size):
    y_true_adverb = output['y_true_adverb'].cpu()
    y_score_antonym = output['y_score_antonym'].cpu()
    y_score = output['y_score'].cpu()
    y_score_antonym_no_act_gt = output['y_score_antonym_no_act_gt'].cpu()
    y_score_no_act_gt = output['y_score_no_act_gt'].cpu()

    assert all(x.shape[0] == dataset_size for x in (y_true_adverb, y_score_antonym, y_score))

    em = {}

    for p, ysa, ys in zip(('', 'no_act_gt/'),
                          (y_score_antonym, y_score_antonym_no_act_gt),
                          (y_score, y_score_no_act_gt)):
        v2a_ant = (torch.argmax(y_true_adverb, dim=1) == torch.argmax(ysa, dim=1)).float().mean().item()
        v2a_all = average_precision_score(y_true_adverb, ys, average='samples')

        a2v_ant = average_precision_score(y_true_adverb, ysa)
        a2v_all = average_precision_score(y_true_adverb, ys)

        a2v_ant_w = average_precision_score(y_true_adverb, ysa, average='weighted')
        a2v_all_w = average_precision_score(y_true_adverb, ys, average='weighted')

        em[f'{p}vid_to_adv_ant'] = v2a_ant
        em[f'{p}vid_to_adv_all'] = v2a_all

        em[f'{p}adv_to_vid_ant'] = a2v_ant
        em[f'{p}adv_to_vid_all'] = a2v_all

        em[f'{p}adv_to_vid_ant_w'] = a2v_ant_w
        em[f'{p}adv_to_vid_all_w'] = a2v_all_w  # TODO rename metrics

    return em


def run(model, train_loader, test_loader, optimiser, criterion, args, evaluators, output_dict):
    log_writer = SummaryWriter(log_dir=output_dict['logs'])
    dump_for = ('test_vid_to_adv_ant', 'test_vid_to_adv_all', 'test_adv_to_vid_ant', 'test_adv_to_vid_all',
                'test_vid_to_action')  # TODO rename metrics

    train_eval = evaluators.get('train', None)
    test_eval = evaluators.get('test', None)
    summary_output_path = output_dict['root'] / 'summary.csv'
    metrics_meters = {}

    for epoch in range(1, args.epochs + 1):
        print('=' * 120)
        print(f'EPOCH {epoch}')
        print('=' * 120)

        if not args.test_only:
            train_loss, train_output, train_metadata = train(train_loader, model, optimiser, criterion, args,
                                                             train_eval, epoch=epoch)
            train_metrics = compute_evaluation_metrics(train_output, args, len(train_loader.dataset))
            log_run(epoch, log_writer, train_metrics, train_loss, 'Train')
        else:
            train_loss = None
            train_metrics = {}

        if epoch == 1 or epoch % args.test_frequency == 0:
            test_loss, test_output, test_metadata = test(model, test_loader, criterion, args, test_eval)
            test_metrics = compute_evaluation_metrics(test_output, args, len(test_loader.dataset))
            log_run(epoch, log_writer, test_metrics, test_loss, 'Test')
        else:
            test_loss = None
            test_output = None
            test_metrics = {}

        summary_dict = dict(epoch=epoch, train_loss=train_loss, test_loss=test_loss)

        for set_, set_loss in zip(('train', 'test'), (train_loss, test_loss)):
            summary_dict[f'{set_}_loss'] = set_loss

        for set_, metric_dict in zip(('train', 'test'), (train_metrics, test_metrics)):
            metric_dict_to_df_row(metric_dict, metrics_meters, set_, summary_dict)

        summary_df = pd.DataFrame([summary_dict])

        if summary_output_path.exists():
            old_summary_df = pd.read_csv(summary_output_path)
            summary_df = pd.concat([old_summary_df, summary_df], ignore_index=True)

        summary_df.to_csv(summary_output_path, index=False)

        for metric, best_metric in metrics_meters.items():
            if best_metric is None or summary_dict.get(metric, None) is None:
                continue

            if summary_dict[metric] > best_metric or args.test_only:
                metrics_meters[metric] = summary_dict[metric]

                if metric in dump_for:
                    best_output_path = output_dict['model_output'] / f'best_{metric}.pth'
                    print(f'Best {metric} so far, dumping model output and state')
                    torch.save(test_output, best_output_path)
                    best_state_path = output_dict['model_state'] / f'best_{metric}.pth'
                    state_dict = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) \
                        else model.state_dict()

                    if 'glove_emb' in state_dict:
                        state_dict.pop('glove_emb')

                    to_pop_s3d = [k for k in state_dict.keys() if k.startswith('s3d')]

                    for k in to_pop_s3d:
                        state_dict.pop(k)

                    state = dict(model_state=state_dict, optimiser=optimiser.state_dict(), epoch=epoch)
                    torch.save(state, best_state_path)

        if args.test_only:
            break

    log_writer.close()


def metric_dict_to_df_row(metric_dict, metrics_meters, set_, summary_dict):
    for k, v in metric_dict.items():
        if v is None:
            continue
        k = k.replace('/', '_')
        kd = f'{set_}_{k}'
        summary_dict[kd] = v.item() if isinstance(v, torch.Tensor) else v

        if kd not in metrics_meters:
            metrics_meters[kd] = 0


def log_run(epoch, log_writer, metrics, loss, tag):
    print(f'Average {tag} loss: {loss:0.4f}')
    log_writer.add_scalar(f'{tag}/loss', loss, epoch)

    for k, v in metrics.items():
        k = k.split('/')

        if len(k) == 1:
            k = k[0]
            k_tag = tag
        else:
            kt = k[:-1]
            k = k[-1]
            k_tag = tag + '_' + '_'.join(kt)

        log_writer.add_scalar(f'{k_tag}/{k}', v, epoch)
        print(f'{k_tag} {k}: {v:0.4f}')


def setup_output(args):
    run_tags = tuple(args.run_tags)
    sub_paths = ('logs', 'model_output', 'model_state')

    if args.tag is not None:
        run_tags += ('tag', 's3d_video_f')

    run_id = ';'.join([f'{attr}={getattr(args, attr)}' for attr in run_tags])
    output_path = args.output_path / args.model / args.modality / args.labels / run_id

    count = 0

    while output_path.exists():
        output_path = output_path.parent / f'{run_id}.{count}'
        count += 1

    output_dict = {}

    for p in sub_paths:
        sp = output_path / p
        sp.mkdir(parents=True, exist_ok=True)
        output_dict[p] = sp

    output_dict['root'] = output_path

    return output_dict, run_id


def setup_evaluator(train_dataset, test_dataset):
    return dict(train=Evaluator(train_dataset), test=Evaluator(test_dataset))


def main():
    parser = create_parser()
    args = parser.parse_args()
    assert not (args.fixed_d and args.cls_variant)
    output_dict, run_id = setup_output(args)
    train_loader, test_loader = setup_data(args)
    evaluators = setup_evaluator(train_loader.dataset, test_loader.dataset)
    model = setup_model(train_loader.dataset, args)
    criterion = setup_criterion(args, train_loader.dataset)
    optimiser = setup_optimiser(model, args)
    run(model, train_loader, test_loader, optimiser, criterion, args, evaluators, output_dict)


if __name__ == '__main__':
    main()
