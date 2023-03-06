# Calculates delta using verb-adverb textual relationships. See paper for more details

import torch
import numpy as np
from torch import pairwise_distance

from s3dg import S3D


class TextEncoder(object):
    __encoder__ = None

    @classmethod
    def get_text_embeddings(cls, args, words):
        assert isinstance(words, list)

        if cls.__encoder__ is None:
            print(f'Creating S3D text encoder')
            init_dict_path = args.s3d_init_folder / 's3d_dict.npy'
            s3d_checkpoint_path = args.s3d_init_folder / 's3d_howto100m.pth'
            print(f'Loading S3D weights from {args.s3d_init_folder}')
            original_classes = 512
            s3d = S3D(init_dict_path, num_classes=original_classes)
            s3d.load_state_dict(torch.load(s3d_checkpoint_path))
            encoder = s3d.text_module
            encoder = encoder.cuda()
            cls.__encoder__ = encoder

        embeddings = cls.__encoder__(words, to_gpu=True)['text_embedding']

        return embeddings

    @classmethod
    def compute_delta(cls, args, dataset_data, antonyms, normalise=False, norm_by_col=False,
                      no_ant=False, absolute_cos_sim=True):
        adverbs, verbs, pairs = dataset_data['adverbs'], dataset_data['verbs'], dataset_data['pairs']
        adverb2idx, verb2idx = dataset_data['adverb2idx'], dataset_data['verb2idx']
        m_delta = np.full((len(adverbs), len(verbs)), np.nan)
        m_d = np.full((len(adverbs), len(verbs)), np.nan)
        index = {}
        x = 0
        cos_sim = torch.nn.CosineSimilarity()
        d_dict = {}
        delta_dict = {}

        for adv, ant in antonyms.items():
            if ant in index:
                continue

            index[adv] = x
            x += 1
            index[ant] = x
            x += 1

        for v, vi in verb2idx.items():
            ve = TextEncoder.get_text_embeddings(args, [v])
            d_dict[v] = {}
            delta_dict[v] = {}

            for a, ai in adverb2idx.items():
                na = antonyms[a]
                s = f'{v} {a}'
                ns = v if no_ant else f'{v} {na}'

                se = TextEncoder.get_text_embeddings(args, [s])
                nse = TextEncoder.get_text_embeddings(args, [ns])
                ae = TextEncoder.get_text_embeddings(args, [a])

                d = pairwise_distance(se, nse)
                scale = cos_sim(ve, ae)

                if absolute_cos_sim and scale < 0:
                    scale = abs(scale)

                delta = d * scale

                m_d[index[a], vi] = d
                m_delta[index[a], vi] = delta

        if normalise:
            if norm_by_col:
                mmax = np.nanmax(m_delta, axis=0)

                for col in range(m_delta.shape[1]):
                    m_delta[:, col] /= mmax[col]
            else:
                m_delta = m_delta / np.nanmax(m_delta)

        for v, vi in verb2idx.items():
            for a, ai in adverb2idx.items():
                d_dict[v][a] = m_d[index[a], vi]
                delta_dict[v][a] = m_delta[index[a], vi]

        return m_delta, m_d, delta_dict, d_dict, list(verb2idx.keys()), list(adverb2idx.keys()), antonyms, index
