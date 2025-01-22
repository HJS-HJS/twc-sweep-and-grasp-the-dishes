import os
import torch
import numpy as np

class Angle():
    def __init__(self, start, end):
        self.start = start
        self.end = end

        if start > end:
            self.end += 2 * np.pi

    def resize(self, size):
        self.start += size
        self.end += size

    def add_margin(self, margin):
        self.start -= margin
        self.end += margin

    @property
    def center(self):
        return (self.start + self.end) / 2

    @property
    def size(self):
        return self.end - self.start
    
    @staticmethod
    def sum(a, b):
        diff = (b.center - a.center)%(2 * np.pi)
        if diff > np.pi:
            a, b = b, a
            diff = 2 * np.pi - diff

        offset = a.center + diff - b.center
        b.resize(offset)
        start = np.min([a.start, b.start])
        end = np.max([a.end, b.end])

        test = Angle(start, end)

        b.resize(-offset)

        return test
    
    @staticmethod
    def distance(a, b):
        diff = (b.center - a.center)%(2 * np.pi)
        if diff > np.pi:
            a, b = b, a
            diff = 2 * np.pi - diff
            # print("a, b changed")

        # center = (a.center + diff / 2) % (2 * np.pi)
        offset = a.center + diff - b.center
        b.resize(offset)
        return b.start - a.end
    
def load_model(network:torch.nn.Module, save_dir:str, name:str, episode:int):
    if episode is None:
        folders = [f for f in os.listdir(save_dir) if os.path.isdir(os.path.join(save_dir, f))]
        numbered_folders = [int(folder) for folder in folders if folder.isdigit()]
        episode = max(numbered_folders)

    _save_dir = save_dir + "/" + str(episode) + "/"
    save_path = os.path.join(_save_dir, f"{str(name)}.pth")
    if os.path.exists(_save_dir):
        state_dict = torch.load(save_path)
        network.load_state_dict(state_dict)
        print("RL Model loaded successfully")
    else:
        print(f"\tNo saved RL model found {save_path}")
    return network