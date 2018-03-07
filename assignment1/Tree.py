class MultiSplitTree:
    def __init__(self):
        self.isTree = False
        self.entropy = None
        self.mostCommonLabel = None
        self.attribute = None
        self.value = None
        self.branches = []
        self.isLeaf = False
        self.label = None


class BinarySplitTree:
    def __init__(self):
        self.isTree = False
        self.entropy = None
        self.mostCommonLabel = None
        self.attribute = None
        self.value = None
        self.trueBranch = None
        self.falseBranch = None
        self.isLeaf = False
        self.label = None
