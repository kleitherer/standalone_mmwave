#!/usr/bin/env python3
"""Reassemble DCA1000 UDP packets into full radar frames."""

import numpy as np


class FrameBuffer:
    def __init__(self, capacity: int, frame_size: int):
        if capacity % frame_size != 0:
            raise ValueError("capacity must be a multiple of frame_size")

        self.buffer = np.zeros(capacity, dtype=np.int8)
        self.frame_size = frame_size
        self.curr_idx = 0
        self.last_seqn = 0
        self.buffer_size = capacity

    def pad_zeros(self, n_msgs: int, msg_size: int) -> None:
        total_size = n_msgs * msg_size
        if self.curr_idx + total_size > self.buffer_size:
            self.buffer[self.curr_idx:] = 0
            self.buffer[: (self.curr_idx + total_size) % self.buffer_size] = 0
            self.curr_idx = (self.curr_idx + total_size) % self.buffer_size
        else:
            self.buffer[self.curr_idx : self.curr_idx + total_size] = 0

    def add_msg(self, seqn: int, msg: bytes):
        msg_arr = np.frombuffer(msg, np.int8)

        if seqn > self.last_seqn + 1:
            self.pad_zeros(seqn - self.last_seqn - 1, msg_arr.size)

        self.last_seqn = seqn

        if self.curr_idx + msg_arr.size > self.buffer_size:
            self.buffer[self.curr_idx :] = msg_arr[: self.buffer_size - self.curr_idx]
            self.buffer[: msg_arr.size - (self.buffer_size - self.curr_idx)] = msg_arr[
                self.buffer_size - self.curr_idx :
            ]
        else:
            self.buffer[self.curr_idx : self.curr_idx + msg_arr.size] = msg_arr

        old_frame_idx = self.curr_idx // self.frame_size
        self.curr_idx = (self.curr_idx + msg_arr.size) % self.buffer_size
        new_frame_idx = self.curr_idx // self.frame_size

        frame_view = self.buffer[
            old_frame_idx * self.frame_size : (old_frame_idx + 1) * self.frame_size
        ].view(np.int16)
        return frame_view, old_frame_idx != new_frame_idx
