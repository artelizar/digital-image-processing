"""
Implementation of the disjoint-set
"""


class DisjointSet:
    """
    Disjoint-set class implementation as an union–find data structure
    """

    def __init__(self):
        self.parents = []
        self.ranks = []

    def make_set(self):
        """
        Creating a new set
        """
        self.parents.append(len(self.parents))
        self.ranks.append(0)

        return len(self.parents) - 1

    def find_set(self, x):
        """
        Returns the root of the set containing an element x
        """
        if self.parents[x] != x:
            # Path compression
            self.parents[x] = self.find_set(self.parents[x])

        return self.parents[x]

    def union_sets(self, x, y):
        """
        Union of sets containing elements x and y
        """
        x_root, y_root = self.find_set(x), self.find_set(y)
        if x_root != y_root:
            # Union by rank
            if self.ranks[x_root] < self.ranks[y_root]:
                x_root, y_root = y_root, x_root

            self.parents[y_root] = x_root
            if self.ranks[x_root] == self.ranks[y_root]:
                self.ranks[x_root] += 1

        return x_root
