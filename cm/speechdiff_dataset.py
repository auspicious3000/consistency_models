# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import logging
import os
import sys
from typing import Any, List, Optional, Union, Dict

import numpy as np
import soundfile as sf
import pickle

import torch
import torch.nn.functional as F
from fairseq.data import data_utils
from fairseq.data.fairseq_dataset import FairseqDataset
from fairseq.tokenizer import tokenize_line

import io
from fairseq.pdb import set_trace

logger = logging.getLogger(__name__)


def get_shard_range(tot, nshard, rank):
    assert rank < nshard and rank >= 0, f"invaid rank/nshard {rank}/{nshard}"
    start = round(tot / nshard * rank)
    end = round(tot / nshard * (rank + 1))
    assert start < end, f"start={start}, end={end}"
    logger.info(
        f"rank {rank} of {nshard}, process {end-start} "
        f"({start}-{end}) out of {tot}"
    )
    return start, end


def load_audio(manifest_path, max_keep, min_keep, split, nshard, rank):
    n_long, n_short = 0, 0
    names, inds, sizes, speakers = [], [], [], []
    with open(manifest_path) as f:
        root = f.readline().strip()
        lines = [line.rstrip() for line in f]
        if nshard and rank:
            start, end = get_shard_range(len(lines), nshard, rank)
            lines = lines[start:end]
        for ind, line in enumerate(lines):
            items = line.strip().split("\t")
            assert len(items) == 2, line
            sz = int(items[1])
            if min_keep is not None and sz < min_keep:
                n_short += 1
            elif max_keep is not None and sz > max_keep:
                n_long += 1
            else:
                spk = items[0].split('/')[1]
                names.append(items[0])
                inds.append(ind)
                sizes.append(sz)
                speakers.append(spk)
    tot = ind + 1
    logger.info(
        (
            f"split={split}, max_keep={max_keep}, min_keep={min_keep}, "
            f"loaded {len(names)}, skipped {n_short} short and {n_long} long, "
            f"longest-loaded={max(sizes)}, shortest-loaded={min(sizes)}"
        )
    )
    return root, names, inds, tot, sizes, list(set(speakers))


def load_label(label_path, inds, tot, nshard, rank):
    with open(label_path) as f:
        labels = [line.rstrip() for line in f]
        if nshard and rank:
            start, end = get_shard_range(len(labels), nshard, rank)
            labels = labels[start:end]
        assert (
            len(labels) == tot
        ), f"number of labels does not match ({len(labels)} != {tot})"
        labels = [labels[i] for i in inds]
    return labels


def load_label_offset(label_path, inds, tot, nshard, rank):
    with open(label_path) as f:
        lines = [line for line in f]
        code_offset = 0
        if nshard and rank:
            start, end = get_shard_range(len(lines), nshard, rank)
            code_offset = sum([len(line.encode("utf-8")) for line in lines[:start]])
            lines = lines[start:end]
        code_lengths = [len(line.encode("utf-8")) for line in lines]
        assert (
            len(code_lengths) == tot
        ), f"number of labels does not match ({len(code_lengths)} != {tot})"
        offsets = list(itertools.accumulate([code_offset] + code_lengths))
        offsets = [(offsets[i], offsets[i + 1]) for i in inds]
    return offsets


def verify_label_lengths(
    audio_sizes,
    audio_rate,
    label_path,
    label_rate,
    inds,
    tot,
    tol=0.1,  # tolerance in seconds
    nshard=None,
    rank=None,
):
    if label_rate < 0:
        logger.info(f"{label_path} is sequence label. skipped")
        return

    with open(label_path) as f:
        lines = [line for line in f]
        if nshard and rank:
            start, end = get_shard_range(len(lines), nshard, rank)
            lines = lines[start:end]
        lengths = [len(line.rstrip().split()) for line in lines]
        assert len(lengths) == tot
        lengths = [lengths[i] for i in inds]
    num_invalid = 0
    for i, ind in enumerate(inds):
        dur_from_audio = audio_sizes[i] / audio_rate
        dur_from_label = lengths[i] / label_rate
        if abs(dur_from_audio - dur_from_label) > tol:
            logger.warning(
                (
                    f"audio and label duration differ too much "
                    f"(|{dur_from_audio} - {dur_from_label}| > {tol}) "
                    f"in line {ind+1} of {label_path}. Check if `label_rate` "
                    f"is correctly set (currently {label_rate}). "
                    f"num. of samples = {audio_sizes[i]}; "
                    f"label length = {lengths[i]}"
                )
            )
            num_invalid += 1
    if num_invalid > 0:
        logger.warning(
            f"total {num_invalid} (audio, label) pairs with mismatched lengths"
        )
        
        
def sequence_mask(lengths, max_len=None):
    """
    Creates a boolean mask from sequence lengths.
    Masked == True
    """
    batch_size = lengths.numel()
    max_len = max_len or lengths.max()
    return ~(torch.arange(0, max_len, device=lengths.device)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1)))


def collate_tensors(
    values,
    pad_idx,
    pad_to_length=None,
    pad_to_multiple=1,
    pad_to_bsz=None,
):
    """Convert a list of 2d tensors into a padded 2d tensor."""
    size = max(v.size(0) for v in values)
    size = size if pad_to_length is None else max(size, pad_to_length)
    if pad_to_multiple != 1 and size % pad_to_multiple != 0:
        size = int(((size - 0.1) // pad_to_multiple + 1) * pad_to_multiple)

    batch_size = len(values) if pad_to_bsz is None else max(len(values), pad_to_bsz)
    res = values[0].new(batch_size, size, values[0].size(1)).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(v, res[i][: len(v)])
    return res


class LabelEncoder:
    def __call__(self, label):
        words = tokenize_line(label)
        nwords = len(words)
        ids = torch.IntTensor(nwords)
        for i, word in enumerate(words):
            ids[i] = int(word)
        return ids


class SpeechDiffDataset(FairseqDataset):
    def __init__(self,
        manifest_path: str,
        sample_rate: float,
        feat_root: str,
        label_path: str,
        label_rate: float,
        max_keep_sample_size: Optional[int] = None,
        min_keep_sample_size: Optional[int] = None,
        max_sample_size: Optional[int] = None,
        min_sample_size: Optional[int] = None,
        nshard: Optional[int] = None,
        rank: Optional[int] = None,
        store_labels: bool = False,
        spk2info: str = '/',
        random_speaker: Optional[float] = 0.0,
        random_resample: Optional[float] = 0.0,
        random_shift: Optional[float] = 0.0,
        rng_seed: Optional[int] = None,
    ):
        self.split = manifest_path.split('/')[-1][:-4]
        assert self.split in ['train', 'valid']
        with open(spk2info, "rb") as f:
            spk2info = pickle.load(f)
        self.spk2info = spk2info['train']
        self.speakers = list(self.spk2info.keys())
        
        self.rng = np.random.default_rng(rng_seed)
        
        self.audio_root, self.audio_names, inds, tot, self.sizes, _ = load_audio(
            manifest_path, max_keep_sample_size, min_keep_sample_size, 
            self.split, nshard, rank,
        )
        self.sample_rate = sample_rate
        self.random_speaker = random_speaker
        self.random_resample = random_resample
        self.random_shift = random_shift
        
        self.feat_root = feat_root
        self.frame_rate = sample_rate // label_rate
                
        self.label_processors = LabelEncoder()
        self.label_rate = label_rate
        self.store_labels = store_labels
        if store_labels:
            self.label_list = load_label(label_path, inds, tot, nshard, rank)
        else:
            self.label_path = label_path
            self.label_offsets_list = load_label_offset(label_path, inds, tot, nshard, rank)
        
        verify_label_lengths(
            self.sizes, sample_rate, label_path, label_rate, inds, tot, nshard=nshard, rank=rank
        )

        self.max_sample_size = (
            max_sample_size if max_sample_size is not None else sys.maxsize
        )
        
        self.min_sample_size = min_sample_size
        if self.min_sample_size is not None:
            assert self.min_sample_size >= min_keep_sample_size
            logger.info(
                f"min_sample_size={self.min_sample_size}"
            )
        
        logger.info(
            f"max_sample_size={self.max_sample_size}"
        )
        
    def get_audio(self, index):
        import soundfile as sf

        wav_path = os.path.join(self.audio_root, self.audio_names[index])
        wav, cur_sample_rate = sf.read(wav_path)
        wav = torch.from_numpy(wav).float()
        wav = self.postprocess(wav, cur_sample_rate)
        
        return wav
    
    def get_feat(self, index):
        feat_name = self.audio_names[index].split('.')[0] + '.npy'
        feat_path = os.path.join(self.feat_root, feat_name)
        feat = np.load(feat_path)
        feat = torch.from_numpy(feat).float()
        return feat

    def get_label(self, index):
        if self.store_labels:
            label = self.label_list[index]
        else:
            with open(self.label_path) as f:
                offset_s, offset_e = self.label_offsets_list[index]
                f.seek(offset_s)
                label = f.read(offset_e - offset_s)
        if self.label_processors is not None:
            label = self.label_processors(label)
        return label + 2 # reserve 0, 1 for special purpose

    def __getitem__(self, index):
        wav = self.get_audio(index)
        feat = self.get_feat(index)
        label = self.get_label(index)
        
        assert len(feat) == len(label), len(feat)-len(label)
        
        fileName = self.audio_names[index]
        spk_src = fileName.split('/')[1]
        
        spk_emb_src, _ = self.spk2info[spk_src]
        spk_emb_src = torch.from_numpy(spk_emb_src).float()
        
        if self.rng.random() < self.random_speaker:
            spk_tgt = self.rng.choice(self.speakers)
        else:
            spk_tgt = spk_src
        spk_emb_tgt, _ = self.spk2info[spk_tgt]
        spk_emb_tgt = torch.from_numpy(spk_emb_tgt).float()
        
        return {"id": index, "wav": wav, "feat": feat, "label": label,
                "spk_emb_src": spk_emb_src, "spk_emb_tgt": spk_emb_tgt}

    def __len__(self):
        return len(self.sizes)
    
    def crop_to_max_size(self, wav, target_size):
        size = len(wav)
        diff = size - target_size
        if diff <= 0:
            return wav, 0, size
        
        if self.min_sample_size is not None:
            target_size = self.rng.integers(
                self.min_sample_size, 
                self.max_sample_size + 1)
            diff = size - target_size
        #start, end = 0, target_size
        start = self.rng.integers(0, diff + 1)
        end = size - diff + start
        if self.min_sample_size is not None:
            diff = target_size - self.max_sample_size
            audio = torch.cat([wav[start:end], wav.new_full((-diff,), 0.0)])
        else:
            audio = wav[start:end]
        return audio, start, end-start, diff

    def collater(self, samples):
        samples = [s for s in samples if s["wav"] is not None]
        if len(samples) == 0:
            return {}

        audios = [s["wav"] for s in samples]
        
        collated_audios, audio_mask, audio_starts, audio_sizes = \
        self.collater_audio(
            audios
        )
        
        feats = [s["feat"] for s in samples]
        labels = [s["label"] for s in samples]
        
        collated_labels, label_mask, collated_feats, feat_mask = \
        self.collater_frm_label(
                    labels, feats, audio_sizes, audio_starts, self.label_rate, 0
                )
        
        spk_embs_src = [s["spk_emb_src"] for s in samples]
        collated_embs_src = torch.stack(spk_embs_src)
        
        spk_embs_tgt = [s["spk_emb_tgt"] for s in samples]
        collated_embs_tgt = torch.stack(spk_embs_tgt)
          
        net_input = {"audios": collated_audios, "audio_mask": audio_mask,
                     "feats": collated_feats, "feat_mask": feat_mask,
                     "labels": collated_labels, "label_mask": label_mask,
                     "spk_embs_src": collated_embs_src, 
                     "spk_embs_tgt": collated_embs_tgt
        }
        
        cond = {"labels": collated_labels, "label_mask": label_mask,
                "spk_embs_src": collated_embs_src,
        }
                
        batch = {
            "id": torch.LongTensor([s["id"] for s in samples]),
            "net_input": net_input,
        }
        
        return collated_feats, cond, batch
    
    def collater_audio(self, audios):
        collated_audios = audios[0].new_zeros(len(audios), self.max_sample_size)
        padding_mask = (
            torch.BoolTensor(collated_audios.shape).fill_(False)
            # if self.pad_audio else None
        )
        audio_starts = [0 for _ in audios]
        audio_sizes = [len(s) for s in audios]
        for i, audio in enumerate(audios):
            diff = len(audio) - self.max_sample_size
            if diff == 0:
                collated_audios[i] = audio
            elif diff < 0:
                collated_audios[i] = torch.cat([audio, audio.new_full((-diff,), 0.0)])
                padding_mask[i, diff:] = True
            else:
                collated_audios[i], audio_starts[i], audio_sizes[i], diff = self.crop_to_max_size(
                    audio, self.max_sample_size
                )
                if diff < 0:
                    padding_mask[i, diff:] = True
        max_audio_len = max(audio_sizes)
        collated_audios = collated_audios[:, :max_audio_len]
        padding_mask = padding_mask[:, :max_audio_len]
        return collated_audios, padding_mask, audio_starts, audio_sizes
    
    def collater_frm_label(self, targets, feats, audio_sizes, audio_starts, label_rate, pad):
        assert label_rate > 0
        s2f = label_rate / self.sample_rate
        frm_starts = [int(round(s * s2f)) for s in audio_starts]
        frm_sizes = [int(round(audio_size * s2f)) for audio_size in audio_sizes]
        #frm_size_max = int(round(self.max_sample_size * s2f))
        targets_new = []
        feats_new = []
        for t, ft, s, frm_size in zip(targets, feats, frm_starts, frm_sizes):
            lab = t[s : s + frm_size] # for padded audio, s+frm_size includes end
            ft = ft[s : s + frm_size]
            feats_new.append(ft)
            
            if self.rng.random() < self.random_resample:
                lab_unique, counts = lab.unique_consecutive(return_counts=True)
                sf = self.rng.random() > 0.5 # choose slow or fast
                c = self.rng.random() * (0.8-0.5) + 0.5 if sf else self.rng.random() * (1.5-1.2) + 1.2
                k = self.rng.random((len(counts),)) * 0.1 - 0.04 + c # perturb around rate multiplier
                r = torch.round(counts * k).to(torch.int32)
                r[r==0] = 1 # preserv at least 1 state
                lab = lab_unique.repeat_interleave(r)
                
            if self.rng.random() < self.random_shift:
                len_shift = int(round(0.25*self.rng.random()*len(lab)))
                sil = self.rng.choice([442], size=(len_shift,)).astype(np.int32)
                sil = torch.from_numpy(sil)
                lab = torch.cat([sil, lab])
            
            targets_new.append(lab)
        
        logger.debug(f"audio_starts={audio_starts}")
        logger.debug(f"frame_starts={frm_starts}")
        logger.debug(f"frame_size={frm_size}")

        len_targets = torch.LongTensor([len(t) for t in targets_new])
        targets = data_utils.collate_tokens(targets_new, pad_idx=pad, 
                                            left_pad=False)
        mask_targets = sequence_mask(len_targets)
        
        len_feats = torch.LongTensor([len(ft) for ft in feats_new])
        feats = collate_tensors(feats_new, pad_idx=pad)
        mask_feats = sequence_mask(len_feats)
        
        return targets, mask_targets, feats, mask_feats

    def num_tokens(self, index):
        return self.size(index)

    def size(self, index):
        return min(self.sizes[index], self.max_sample_size)

    def postprocess(self, wav, cur_sample_rate):
        if wav.dim() == 2:
            wav = wav.mean(-1)
        assert wav.dim() == 1, wav.dim()

        if cur_sample_rate != self.sample_rate:
            raise Exception(f"sr {cur_sample_rate} != {self.sample_rate}")

        return wav
    
    
from mpi4py import MPI
from torch.utils.data import DataLoader
    
    
def load_data(
    *,
    manifest_path,
    feat_root,
    label_path,
    spk2info,
    batch_size,
    deterministic=False,
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    """    
    dataset = SpeechDiffDataset(manifest_path=manifest_path,
                                sample_rate=16000,
                                feat_root=feat_root,
                                label_path=label_path,
                                label_rate=50,
                                max_keep_sample_size=None,
                                min_keep_sample_size=32000,
                                max_sample_size=96000,
                                min_sample_size=32000,
                                nshard=MPI.COMM_WORLD.Get_size(),
                                rank=MPI.COMM_WORLD.Get_rank(),
                                store_labels=False,
                                spk2info=spk2info,
                                random_speaker=0.0,
                                random_resample=0.0,
                                random_shift=0.0)
    
    def worker_init_fn(x):
        return np.random.seed((torch.initial_seed()) % (2**32)) 
    
    if deterministic:
        loader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=8,
                            drop_last=True,
                            pin_memory=True,
                            worker_init_fn=worker_init_fn,
                            collate_fn=dataset.collater)
    else:
        loader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=8,
                            drop_last=True,
                            pin_memory=True,
                            worker_init_fn=worker_init_fn,
                            collate_fn=dataset.collater)
    while True:
        yield from loader