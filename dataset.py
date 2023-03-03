import numpy as np
import torch
from torch.nn.utils.rnn import pad_packed_sequence, pack_sequence


def collate_variable_length_seq(batch, padding_value=0, modality='rgb'):
    data = [[], []] if modality == 'rgb_flow' else []
    labels = {}
    metadata = {}

    for item in batch:
        d, l, m = item

        if modality == 'rgb_flow':
            rgb, flow = d
            data[0].append(rgb)
            data[1].append(flow)
        else:
            data.append(d)

        for stuff, stuff_batch in zip((labels, metadata), (l, m)):
            if len(stuff) == 0:
                for k in stuff_batch.keys():
                    stuff[k] = []

            for k, v in stuff_batch.items():
                stuff[k].append(v)

    if modality == 'rgb_flow':
        padded_sequence = []

        for modality_data in data:
            mod_seq = pack_sequence(modality_data, enforce_sorted=False)
            mod_pad_seq, _ = pad_packed_sequence(mod_seq, batch_first=True, padding_value=padding_value)
            padded_sequence.append(mod_pad_seq)

        padded_sequence = tuple(padded_sequence)
    else:
        if isinstance(data[0], dict):
            keys = data[0].keys()
            padded_sequence = {}

            for k in keys:
                l = [x[k] for x in data]
                seq = pack_sequence(l, enforce_sorted=False)
                padded_sequence[k], _ = pad_packed_sequence(seq, batch_first=True, padding_value=padding_value)
        else:
            sequence = pack_sequence(data, enforce_sorted=False)
            padded_sequence, _ = pad_packed_sequence(sequence, batch_first=True, padding_value=padding_value)

    labels = {k: torch.LongTensor(v) for k, v in labels.items()}

    return padded_sequence, labels, metadata


class Dataset(torch.utils.data.Dataset):
    def __init__(self, df,  antonyms_df, features_dict, pickle_data, feature_dim, no_antonyms=False):
        self.df = df
        self.random_generator = np.random.default_rng()
        self.no_antonyms = no_antonyms
        self.antonyms = {t.adverb: t.antonym for t in antonyms_df.itertuples()}

        self.feature_modalities = list(features_dict.keys())
        self.features = features_dict['features']
        self.metadata =  features_dict['metadata']
        self.feature_dim = feature_dim

        self.adverbs = pickle_data['adverbs']
        self.verbs = pickle_data['verbs']
        self.pairs = pickle_data['pairs']
        self.adverb2idx = pickle_data['adverb2idx']
        self.verb2idx = pickle_data['verb2idx']
        self.idx2verb = pickle_data['idx2verb']
        self.idx2adverb = {v: k for k, v in self.adverb2idx.items()}
        self.data_pickle = pickle_data

    def __len__(self):
        return len(self.df)

    def get_verb_adv_pair_idx(self, labels):
        v_str = [self.idx2verb[x.item() if isinstance(x, torch.Tensor) else x] for x in labels['verb']]
        a_str = [self.idx2adverb[x.item() if isinstance(x, torch.Tensor) else x] for x in labels['adverb']]
        va_idx = [self.pairs.index((a, v)) for a, v in zip(a_str, v_str)]
        return va_idx

    def get_adverb_with_verb(self, verb):
        verb = verb.item() if isinstance(verb, torch.Tensor) else verb
        verb_str = verb if isinstance(verb, str) else self.idx2verb[verb]
        return [a for a, v in self.pairs if v == verb_str]

    def get_verb_with_adverb_mask(self, adverb):
        adverb = adverb.item() if isinstance(adverb, torch.Tensor) else adverb
        adverb_str = adverb if isinstance(adverb, str) else self.idx2adverb[adverb]
        return [a == adverb_str for a, v in self.pairs]

    def __getitem__(self, item):
        if isinstance(item, torch.Tensor):
            item = item.item()

        segment = self.df.iloc[item]

        verb_label = segment.verb_label
        adverb_label = segment.adverb_label
        adverb = segment.clustered_adverb

        labels = dict(verb=verb_label, adverb=adverb_label)
        metadata = {k: getattr(segment, k) for k in ('seg_id', 'start_time', 'end_time', 'clustered_adverb',
                                                     'clustered_verb') if hasattr(segment, k)}

        data, frame_samples = self.load_features(segment)
        metadata['frame_samples'] = frame_samples

        if self.no_antonyms:
            pool = [ai for aa, ai in self.adverb2idx.items() if aa != adverb]
            neg_adverb = np.random.choice(pool, 1)
        else:
            neg_adverb = self.adverb2idx[self.antonyms[adverb]]

        assert adverb_label != neg_adverb
        metadata['negative_adverb'] = neg_adverb

        return data, labels, metadata

    def load_features(self, segment):
        uid = self.get_seg_id(segment)
        features = self.features[uid]
        frame_samples = self.metadata[uid]['frame_samples'].squeeze()
        return features, frame_samples

    @staticmethod
    def get_seg_id(segment, to_str=True):
        seg_id = segment['seg_id'] if isinstance(segment, dict) else getattr(segment, 'seg_id')

        if to_str:
            return str(seg_id)
        else:
            return seg_id

