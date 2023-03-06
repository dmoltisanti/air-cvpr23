import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
from evaluator import Evaluator
from model import RegClsAdverbModel
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import Dataset, collate_variable_length_seq, get_verbs_adverbs_pairs


def create_parser():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('train_df_path', type=Path)
    parser.add_argument('test_df_path', type=Path)
    parser.add_argument('antonyms_df', type=Path)
    parser.add_argument('features_path', type=Path)
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
    parser.add_argument('--run_tags', default=['lr', 'train_batch', 'dropout', 's3d_video_f'], action='append')
    parser.add_argument('--tag', default=None, type=str)
    parser.add_argument('--train_only', action='store_true')
    parser.add_argument('--test_only', action='store_true')
    parser.add_argument('--test_frequency', default=10, type=int)
    parser.add_argument('--hidden_units', default='512,512,512', type=str)
    parser.add_argument('--s3d_init_folder', type=Path, default='./s3d')  # TODO clone repo and download stuff in the setup
    parser.add_argument('--s3d_video_f', default='s3d_features', type=str)
    parser.add_argument('--text_emb_dim', default=512, type=int)

    parser.add_argument('--no_antonyms', action='store_true')
    parser.add_argument('--fixed_d', action='store_true')
    parser.add_argument('--cls_variant', action='store_true')

    return parser


def setup_data(args):
    train_df = pd.read_csv(args.train_df_path)
    test_df = pd.read_csv(args.test_df_path)
    antonyms_df = pd.read_csv(args.antonyms_df)
    dataset_data = get_verbs_adverbs_pairs(train_df, test_df)

    feature_dim = 512 if args.s3d_video_f == 'video_embedding_joint_space' else 1024
    collate_fn = collate_variable_length_seq

    features_train, _ = load_features(args, 'train')
    features_test, _ = load_features(args, 'test')

    train_dataset = Dataset(train_df, antonyms_df, features_train, dataset_data, feature_dim,
                            no_antonyms=args.no_antonyms)

    train_loader = DataLoader(train_dataset, batch_size=args.train_batch, shuffle=True,
                              pin_memory=True, num_workers=args.train_workers, collate_fn=collate_fn)

    test_dataset = Dataset(test_df, antonyms_df, features_test, dataset_data, feature_dim,
                           no_antonyms=args.no_antonyms)

    test_loader = DataLoader(test_dataset, batch_size=args.test_batch, shuffle=False, pin_memory=True,
                             num_workers=args.test_workers, collate_fn=collate_fn)

    return train_loader, test_loader


def load_features(args, set_):
    feature_dict = {}
    features_path = args.features_path / f'{set_}.pth'
    print(f'Loading features from {features_path}')
    fd = torch.load(features_path, map_location=torch.device('cpu'))
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


def setup_criterion():
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


def train(loader, model, optimiser, criterion, evaluator):
    model.train()
    return run_through_loader(model, loader, criterion, optimiser, evaluator, tag='Training')


def test(model, loader, criterion, evaluator):
    model.eval()

    with torch.no_grad():
        return run_through_loader(model, loader, criterion, None, evaluator, tag='Testing', optimise=False)


def run_through_loader(model, loader, criterion, optimiser, evaluator, tag, optimise=True):
    bar = tqdm(desc=tag, file=sys.stdout, total=len(loader))
    loss_batches = []
    scores = {}
    all_metadata = None

    for x_tuple in loader:
        batch_output, loss, batch_labels, batch_metadata = process_batch(x_tuple, model, criterion, optimise)
        batch_scores = get_scores(batch_output, batch_labels, loader.dataset, evaluator)

        if optimise:
            loss.backward()
            optimiser.step()
            optimiser.zero_grad()

        if loss is not None:
            bar.set_description(f'{tag} loss: {loss.detach().item():0.4f}')
            loss_batches.append(loss.detach().item())

        bar.update()

        if batch_scores is not None:
            for k, o in batch_scores.items():
                if o is None:
                    continue

                if k not in scores:
                    scores[k] = []

                scores[k].append(o.detach())

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

    scores = {k: torch.cat([x.unsqueeze(0) if x.ndim == 1 else x for x in v], dim=0) if v else None
              for k, v in scores.items()}

    return avg_loss, scores, all_metadata


def get_scores(output, labels, dataset, evaluator, stack_scores=False):
    with torch.no_grad():
        predictions = output[1]

        if predictions is None:
            return None

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
                [action_gt_scores[idx][dataset.pairs.index((adv, dataset.idx2verb[verb_gt[idx].item()]))]
                 for adv in dataset.adverbs])

            y_score_antonym[idx] = torch.Tensor(
                [antonym_action_gt_scores[idx][dataset.pairs.index((adv, dataset.idx2verb[verb_gt[idx].item()]))]
                    for adv in dataset.adverbs])

            for ia, a in enumerate(dataset.adverbs):
                mask = dataset.get_verb_with_adverb_mask(a)
                y_score_no_act_gt[idx, ia] = scores_no_act_gt[idx, mask].max()
                y_score_antonym_no_act_gt[idx, ia] = scores_no_act_gt_ant[idx, mask].max()

        out_dict = dict(y_true_adverb=y_true_adverb, y_score_antonym=y_score_antonym,
                        y_score=y_score, scores=scores, scores_no_act_gt=scores_no_act_gt,
                        y_score_no_act_gt=y_score_no_act_gt, y_score_antonym_no_act_gt=y_score_antonym_no_act_gt)

        return out_dict


def process_batch(x_tuple, model, criterion, training):
    features, labels, metadata = x_tuple
    features, labels = to_cuda((features, labels))

    adverbs = labels['adverb']
    verbs = labels['verb']
    neg_adverbs = torch.LongTensor(metadata['negative_adverb']).cuda()
    labels_tuple = (adverbs, verbs, neg_adverbs)
    output = model(features, labels_tuple, training=training)

    loss = criterion(output, labels)
    return output, loss, labels, metadata


def compute_evaluation_metrics(scores, dataset_size):
    if not scores:
        return {}

    y_true_adverb = scores['y_true_adverb'].cpu()
    y_score_antonym = scores['y_score_antonym'].cpu()
    y_score = scores['y_score'].cpu()
    y_score_antonym_no_act_gt = scores['y_score_antonym_no_act_gt'].cpu()
    y_score_no_act_gt = scores['y_score_no_act_gt'].cpu()

    assert all(x.shape[0] == dataset_size for x in (y_true_adverb, y_score_antonym, y_score))
    metrics = {}

    for p, ysa, ys in zip(('', 'no_act_gt/'),
                          (y_score_antonym, y_score_antonym_no_act_gt),
                          (y_score, y_score_no_act_gt)):
        map_w = average_precision_score(y_true_adverb, ys, average='weighted')
        map_m = average_precision_score(y_true_adverb, ys)
        acc_a = (torch.argmax(y_true_adverb, dim=1) == torch.argmax(ysa, dim=1)).float().mean().item()

        metrics[f'{p}map_w'] = map_w
        metrics[f'{p}map_m'] = map_m
        metrics[f'{p}acc_a'] = acc_a

    return metrics


def run(model, train_loader, test_loader, optimiser, criterion, args, evaluators, output_dict):
    log_writer = SummaryWriter(log_dir=output_dict['logs'])
    dump_for = ('test_map_w', 'test_map_m', 'test_acc_a')

    train_eval = evaluators.get('train', None)
    test_eval = evaluators.get('test', None)
    summary_output_path = output_dict['root'] / 'summary.csv'
    metrics_meters = {}

    for epoch in range(1, args.epochs + 1):
        print('=' * 120)
        print(f'EPOCH {epoch}')
        print('=' * 120)

        if not args.test_only:
            train_loss, train_scores, train_metadata = train(train_loader, model, optimiser, criterion, train_eval)
            train_metrics = compute_evaluation_metrics(train_scores, len(train_loader.dataset))
            log_run(epoch, log_writer, train_metrics, train_loss, 'Train')
        else:
            train_loss = None
            train_metrics = {}

        if epoch == 1 or epoch % args.test_frequency == 0:
            test_loss, test_scores, test_metadata = test(model, test_loader, criterion, test_eval)
            test_metrics = compute_evaluation_metrics(test_scores, len(test_loader.dataset))
            log_run(epoch, log_writer, test_metrics, test_loss, 'Test')
        else:
            test_loss = None
            test_scores = None
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
                    torch.save(test_scores, best_output_path)
                    best_state_path = output_dict['model_state'] / f'best_{metric}.pth'
                    state_dict = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) \
                        else model.state_dict()

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


def setup_run_output(args):
    run_tags = tuple(args.run_tags)
    sub_paths = ('logs', 'model_output', 'model_state')

    if args.tag is not None:
        run_tags += ('tag',)

    run_id = ';'.join([f'{attr}={getattr(args, attr)}' for attr in run_tags])

    if args.cls_variant:
        variant = 'cls'
    elif args.fixed_d:
        variant = 'reg_fixed_d'
    else:
        variant = 'reg'

    if args.no_antonyms:
        variant += '_no_antonyms'

    output_path = args.output_path / variant / run_id

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
    assert not (args.no_antonyms and args.cls_variant)
    output_dict, run_id = setup_run_output(args)
    train_loader, test_loader = setup_data(args)
    evaluators = setup_evaluator(train_loader.dataset, test_loader.dataset)
    model = setup_model(train_loader.dataset, args)
    criterion = setup_criterion()
    optimiser = setup_optimiser(model, args)
    run(model, train_loader, test_loader, optimiser, criterion, args, evaluators, output_dict)


if __name__ == '__main__':
    main()
