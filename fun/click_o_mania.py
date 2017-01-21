#!/usr/bin/env python
import itertools


def get_comp(graph, node):
    """Returns a set of nodes in this node's component"""
    unexplored = set(node)
    explored = set()
    while unexplored:
        node = unexplored.pop()
        explored.add(node)
        new_nbrs = graph.edges[node] - explored
        unexplored.update(new_nbrs)
    return explored


class Graph(object):
    def __init__(self, nodes):
        self.nodes = nodes
        # {node -> set(node)}
        self.edges = {n:set() for n in nodes}

    def add_edge(self, n1, n2):
        self.edges[n1].add(n2)
        self.edges[n2].add(n1)

    def connected_comps(self):
        """Returns a [set(node)]"""
        comps = []
        unvisited = set(self.nodes)
        while unvisited:
            # Start new component if necessary
            node = unvisited.pop()
            comp = get_comp(self, node)
            unvisited.difference_update(comp)
            comps.append(comp)
        return comps

    def iter_comps(self):
        for c in self.connected_comps():
            if len(c) == 1 and c[0].len == 1:
                continue
            yield list(c)


class Vector(object):
    def __init__(self, col, top_row, color, length):
        self.col = col
        self.top_row = top_row
        self.color = color
        self.len = length

    def cell(self):
        return (self.top_row, self.col)

    @property
    def bottom_row(self):
        return self.top_row + self.len - 1

    def intersects(self, other):
        if self.color != other.color:
            return False
        if abs(self.col - other.col) != 1:
            return False
        if self.top_row > other.bottom_row:
            return False
        if self.bottom_row < other.top_row:
            return False
        return True


def grid_to_vecs(grid):
    # each column is a list of vectors starting from the top
    columns = [[] for _ in xrange(num_cols)]
    for row_idx, row in enumerate(grid):
        for col_idx, color in enumerate(row):
            if color == '-':
                continue
            col = columns[col_idx]
            if not col:
                col.append(Vector(col_idx, row, color, 0))
            vec = col[-1]
            if vec.color == color:
                # extend this vector
                vec.len += 1
            else:
                col.append(Vector(col_idx, row, color, 1))
    return columns


def cols_to_graph(cols):
    g = Graph([vec for col in cols for vec in col])
    for i in xrange(len(cols) - 1):
        c1 = cols[i]
        c2 = cols[i + 1]
        for v1 in c1:
            for v2 in c2:
                if v1.intersects(v2):
                    g.add_edge(v1, v2)
    return g


class Board(object):
    def __init__(self, grid=None, cols=None):
        # each column is a list of vectors starting from the top
        if grid:
            self.cols = grid_to_vecs(grid)
        elif cols:
            self.cols = [col[:] for col in cols]

        #self.graph = cols_to_graph(self.cols)

    def without(self, vecs):
        """Returns a new board with vecs (a set of Vectors) removed."""
        new_cols = [col[:] for col in self.cols]
        vecs = sorted(vecs, lambda v: v.col)
        for col_idx, vs in itertools.groupby(vecs, lambda v: v.col):
            col = new_cols[col_idx]
            for v in vs:
                col.remove(v)
        return Board(cols=new_cols)
            
            
def explore(board, steps=None):
    """Returns a (steps_to_eliminate_most_cells, cells_remaining).
    steps is a list of (row, col) steps taken so far to obtain board."""

    steps = steps or []
    graph = cols_to_graph(board.cols)
    for comp in graph.iter_comps():
        new_board = board.without(comp)
        explore(new_board, steps[:] + [comp[0].cell()])







        


def nextMove(grid):
    board = Board(grid=grid)

    print ""

num_rows = 0
num_cols = 0
if name == '__main__':
    num_rows, num_cols, k = [ int(i) for i in raw_input().strip().split() ] 
    grid = [[i for i in str(raw_input().strip())] for _ in range(num_rows)] 
    nextMove(grid)

