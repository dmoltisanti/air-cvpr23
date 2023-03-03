# copied from Action Modifier: https://github.com/hazeld/action-modifiers/blob/master/model.py

import torch


class Evaluator:
    def __init__(self, dset):
        self.dset = dset
        pairs = [(dset.adverb2idx[adv.strip()], dset.verb2idx[act]) for adv, act in dset.pairs]
        self.pairs = torch.LongTensor(pairs)

        # mask over pairs for ground-truth action given in testing
        action_gt_mask = []
        for _act in dset.verbs:
            mask = [1 if _act == act else 0 for _, act in dset.pairs]  # DM
            action_gt_mask.append(torch.BoolTensor(mask))

        # DM we do the same for adverb
        adverb_gt_mask = []

        for _adv in dset.adverbs:
            mask = [1 if _adv == adv else 0 for adv, _ in dset.pairs]
            adverb_gt_mask.append(torch.BoolTensor(mask))

        self.action_gt_mask = torch.stack(action_gt_mask, 0)
        self.adverb_gt_mask = torch.stack(adverb_gt_mask, 0)

        antonym_mask = []

        for _adv in dset.adverbs:
            mask = [1 if (_adv==adv or _adv==dset.antonyms[adv]) else 0 for adv, act in dset.pairs]
            antonym_mask.append(torch.BoolTensor(mask))

        self.antonym_mask = torch.stack(antonym_mask, 0)

    def get_gt_action_scores(self, scores, action_gt):
        mask = self.action_gt_mask[action_gt]
        action_gt_scores = scores.clone()
        action_gt_scores[~mask] = -1e10
        return action_gt_scores, mask

    # DM
    def get_gt_adverb_scores(self, scores, adverb_gt):
        mask = self.adverb_gt_mask[adverb_gt]
        adverb_gt_scores = scores.clone()
        adverb_gt_scores[~mask] = -1e10
        return adverb_gt_scores, mask

    def get_antonym_scores(self, scores, adverb_gt):
        mask = self.antonym_mask[adverb_gt]
        antonym_scores = scores.clone()
        antonym_scores[~mask] = -1e10
        return antonym_scores

    def get_gt_action_antonym_scores(self, scores, action_gt, adverb_gt):
        mask = self.antonym_mask[adverb_gt] & self.action_gt_mask[action_gt]
        action_gt_antonym_scores = scores.clone()
        action_gt_antonym_scores[~mask] = -1e10
        return action_gt_antonym_scores

    def get_scores(self, scores, action_gt, adverb_gt, cpu=False, stack_scores=True):
        if cpu:
            scores = {k: v.cpu() for k, v in scores.items()}
            action_gt = action_gt.cpu()

        if stack_scores:
            scores = torch.stack([scores[(adv, act)] for adv, act in self.dset.pairs], 1)

        action_gt_scores, action_mask = self.get_gt_action_scores(scores, action_gt)
        adverb_gt_scores, adverb_mask = self.get_gt_adverb_scores(scores, adverb_gt)
        antonym_action_gt_scores = self.get_gt_action_antonym_scores(scores, action_gt, adverb_gt)

        return scores, action_gt_scores, adverb_gt_scores, antonym_action_gt_scores, action_mask, adverb_mask
