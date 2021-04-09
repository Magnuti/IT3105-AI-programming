# Inspired from https://www.geeksforgeeks.org/disjoint-set-data-structures/

class DisjointSet:
    def __init__(self, elements):
        # Constructor to create and
        # initialize sets of n items
        self.rank = {}
        self.parent = {}

        for e in elements:
            self.rank[e] = 1
            self.parent[e] = e

    # Finds the representative of a set of given item x
    def find(self, x):
        if (self.parent[x] != x):
            self.parent[x] = self.find(self.parent[x])

        return self.parent[x]

    # Do union of two sets represented
    # by x and y.
    def union(self, x, y):
        # Find current sets of x and y
        xset = self.find(x)
        yset = self.find(y)

        # If they are already in same set
        if xset == yset:
            return

        # Put smaller ranked item under
        # bigger ranked item if ranks are
        # different
        if self.rank[xset] < self.rank[yset]:
            self.parent[xset] = yset

        elif self.rank[xset] > self.rank[yset]:
            self.parent[yset] = xset

        # If ranks are same, then move y under
        # x (doesn't matter which one goes where)
        # and increment rank of x's tree
        else:
            self.parent[yset] = xset
            self.rank[xset] = self.rank[xset] + 1
