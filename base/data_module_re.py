import torch
import numpy as np

import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

class SeqDataset(torch.utils.data.Dataset):
    def __init__(self, seqs, states, window, *args, **kwargs):
        self.seqs = seqs
        self.states = states
        self.window = window
        super(*args, **kwargs)

    def __getitem__(self, idx):
        seq = self.seqs[idx, ...].reshape(self.window, -1)
        state = self.states[idx]
        return seq, state

    def __len__(self):
        return len(self.seqs)


def gen_segments(joints, states, window_size):
    """_summary_

    Args:
        joints (_type_): _description_
        states (_type_): _description_
        window_size (_type_): _description_

    Returns:
        w_joints: _description_
        w_states: _description_
    """
    w_joint_list = []
    w_states_list = []

    for joint, state in zip(joints, states):
        start_list = np.arange(0, joint.shape[0] - window_size + 1)
        w_seg = np.array([joint[start : start + window_size] for start in start_list])
        w_joint_list.append(w_seg)
        w_states_list.append(np.full((w_seg.shape[0], 1), state))

    return np.concatenate(w_joint_list), np.concatenate(w_states_list)


class dp_dm(pl.LightningDataModule):
    def __init__(
        self,
        joints,
        states,
        batch_size,
        window_size,
        intvl_size,
        num_workers,
        test_size,
        rs,
        subset_size=None,
    ):
        super().__init__()
        self.joints = joints
        self.states = states
        self.batch_size = batch_size
        self.window_size = window_size
        self.itvl_size = intvl_size  ## Unused right now
        self.test_size = test_size
        self.num_workers = num_workers
        self.subset_size = subset_size
        self.rs = rs

    def setup(self, stage=None):
        ## Now the problem is that each segment does not have the same length!!
        ## proposed workflow: 1) cut them into window size lengths and 2) split them up
        windowed_joints, windowed_states = gen_segments(
            self.joints, self.states, self.window_size
        )

        if self.subset_size is not None:
            _, subset_joints, _, subset_states = train_test_split(
                windowed_joints,
                windowed_states,
                test_size=self.subset_size,
                stratify=windowed_states,
                random_state=self.rs,
            )

        train_joints, val_joints, train_states, val_states = train_test_split(
            windowed_joints if self.subset_size is None else subset_joints,
            windowed_states if self.subset_size is None else subset_states,
            test_size=self.test_size,
            stratify=windowed_states if self.subset_size is None else subset_states,
            random_state=self.rs,
        )
        self.train = SeqDataset(
            train_joints.astype(np.float32), train_states, self.window_size
        )
        self.val = SeqDataset(
            val_joints.astype(np.float32), val_states, self.window_size
        )

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=self.num_workers,
        )
