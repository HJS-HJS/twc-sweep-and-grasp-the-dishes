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
    