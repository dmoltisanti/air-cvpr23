# Model class for the CVPR 23 paper: "Learning Action Changes by Measuring Verb-Adverb Textual Relationships"

import torch.nn as nn
import torch
from attention import SDPAttention
from text_encoder import TextEncoder


def build_mlp(channels_in, channels_out, hidden_units, bn=False, dropout=0.0, act_func='relu',
              add_act_func_to_last=False):
    if isinstance(hidden_units, str):
        hidden_units = tuple(int(x) for x in hidden_units.split(',') if x.strip())

    assert not (dropout and bn), 'Choose either BN or dropout'
    assert act_func in ('relu', 'tanh', 'none'), f'Cannot deal with this activation function yet: {act_func}'
    layers = []

    for i in range(len(hidden_units)):
        if i == 0:
            in_ = channels_in
            out_ = hidden_units[i]

            if bn:
                layers.append(nn.BatchNorm1d(in_))
        else:
            in_ = hidden_units[i - 1]
            out_ = hidden_units[i]

        layers.append(nn.Linear(in_, out_))

        if act_func == 'relu':
            layers.append(nn.ReLU())
        elif act_func == 'tanh':
            layers.append(nn.Tanh())
        elif act_func != 'none':
            raise ValueError(f'Unexpected activation function: {act_func}')

        if dropout:
            layers.append(nn.Dropout(dropout))
        elif bn:
            layers.append(nn.BatchNorm1d(out_))

    if hidden_units:
        layers.append(nn.Linear(hidden_units[-1], channels_out))
    else:
        layers.append(nn.Linear(channels_in, channels_out))

    if add_act_func_to_last:
        if act_func == 'relu':
            layers.append(nn.ReLU())
        elif act_func == 'tanh':
            layers.append(nn.Tanh())
        elif act_func != 'none':
            raise ValueError(f'Unexpected activation function: {act_func}')

    return layers


class RegClsAdverbModel(nn.Module):
    def __init__(self, train_dataset, args, text_emb_dim=512):
        super(RegClsAdverbModel, self).__init__()
        assert not (args.fixed_d and args.cls_variant)
        self.train_dataset = train_dataset
        self.args = args
        self.attention = SDPAttention(self.train_dataset.feature_dim, text_emb_dim, text_emb_dim,
                                      text_emb_dim, heads=4, dropout=args.dropout)
        modifier_input = text_emb_dim

        self.n_verbs = len(self.train_dataset.verbs)
        self.n_adverbs = len(self.train_dataset.adverbs)
        self.n_pairs = len(self.train_dataset.pairs)
        layers = build_mlp(modifier_input, self.n_adverbs, args.hidden_units, dropout=args.dropout)

        self.rho = nn.Sequential(*layers)
        self.ce = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()
        text_embeddings_verbs = TextEncoder.get_text_embeddings(args, self.train_dataset.verbs)
        self.verb_embedding = nn.Embedding.from_pretrained(text_embeddings_verbs, freeze=False)

        _, _, delta_dict, d_dict, _, _, _, _ = TextEncoder.compute_delta(args, train_dataset.dataset_data,
                                                                         train_dataset.antonyms,
                                                                         no_ant=args.no_antonyms)
        self.delta_dict = delta_dict
        self.d_dict = d_dict

    def forward(self, features, labels_tuple, training=True):
        video_features = features['s3d_features']
        adverbs, verbs, neg_adverbs = labels_tuple
        query = self.verb_embedding(verbs)
        video_emb, attention_weights = self.attention(video_features, query)

        o = self.rho(video_emb)

        if self.args.cls_variant:
            target = adverbs
            loss = self.ce(o, target)
        else:
            if self.args.fixed_d:
                d = torch.ones(len(adverbs)).cuda()
            else:
                d = [self.delta_dict[self.train_dataset.idx2verb[v.item()]][self.train_dataset.idx2adverb[a.item()]]
                     for v, a in zip(verbs, adverbs)]
                d = torch.Tensor(d).cuda()

            target = self.create_target_from_delta(d, adverbs, neg_adverbs)
            loss = self.mse(o, target)

        if not training:
            predictions = self.get_predictions(video_features)
            predictions_no_act_gt = predictions
            pred_tuple = (predictions, predictions_no_act_gt)
        else:
            pred_tuple = None

        output = [loss, pred_tuple]

        return output

    def create_target_from_delta(self, delta, adverbs, neg_adverbs):
        assert delta.min() > 0  # the loss assumes this to flip the target for antonyms
        batch_size = len(adverbs)
        target = torch.zeros((batch_size, self.n_adverbs)).cuda()
        target.scatter_(1, adverbs.unsqueeze(1), delta.unsqueeze(1))

        if not self.args.no_antonyms:
            target.scatter_(1, neg_adverbs.unsqueeze(1), -delta.unsqueeze(1))

        return target

    def get_predictions(self, video_features, verbs=None):
        assert verbs is None, 'Do not pass verb labels. Predictions scores are later calculated accordingly'
        batch_size = video_features.shape[0]
        pair_scores = torch.zeros((batch_size, self.n_pairs))

        for verb_idx, verb_str in self.train_dataset.idx2verb.items():
            emb_idx = torch.LongTensor([verb_idx]).repeat(batch_size).cuda()
            q = self.verb_embedding(emb_idx)
            video_emb, _ = self.attention(video_features, q)
            adverb_pred = self.rho(video_emb)

            for adv_idx, adv_str in self.train_dataset.idx2adverb.items():
                p_idx = self.train_dataset.get_verb_adv_pair_idx(dict(verb=[verb_idx], adverb=[adv_idx]))
                assert len(p_idx) == 1
                p_idx = p_idx[0]
                pair_scores[:, p_idx] = adverb_pred[:, adv_idx]

        return pair_scores
