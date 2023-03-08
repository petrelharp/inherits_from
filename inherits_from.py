import tskit
import msprime
import numpy as np

ts = msprime.sim_ancestry(5, sequence_length=100, recombination_rate=0.01)

class InheritsFrom:
    def __init__(
        self,
        num_nodes,
        samples,
        nodes_time,
        edges_left,
        edges_right,
        edges_parent,
        edges_child,
        edge_insertion_order,
        edge_removal_order,
        sequence_length,
        verbosity=0,
        strict=False,
    ):
        # virtual root is at num_nodes; virtual samples are beyond that
        N = num_nodes + 1 + len(samples)
        # Quintuply linked tree
        self.parent = np.full(N, -1, dtype=np.int32)
        self.left_sib = np.full(N, -1, dtype=np.int32)
        self.right_sib = np.full(N, -1, dtype=np.int32)
        self.left_child = np.full(N, -1, dtype=np.int32)
        self.right_child = np.full(N, -1, dtype=np.int32)
        # Sample lists refer to sample *index*
        self.num_samples = np.full(N, 0, dtype=np.int32)
        # Edges and indexes
        self.edges_left = edges_left
        self.edges_right = edges_right
        self.edges_parent = edges_parent
        self.edges_child = edges_child
        self.edge_insertion_order = edge_insertion_order
        self.edge_removal_order = edge_removal_order
        self.sequence_length = sequence_length
        self.nodes_time = nodes_time
        self.samples = samples
        self.position = 0
        self.virtual_root = num_nodes
        # we only need x for the focal node, actually
        self.x = np.zeros(N, dtype=np.float64)
        self.out = np.zeros(N, dtype=np.float64)
        self.verbosity = verbosity
        self.focal_node = tskit.NULL

        n = samples.shape[0]
        for j in range(n):
            u = samples[j]
            self.num_samples[u] = 1
            self.insert_root(u)
            self.insert_branch(u, num_nodes + 1 + j)

    def print_state(self, msg):
        print(f"{msg}")
        for j, (x, z, p) in enumerate(zip(self.x, self.out, self.parent)):
            print(j, x, z, p)

    def remove_branch(self, p, c):
        lsib = self.left_sib[c]
        rsib = self.right_sib[c]
        if lsib == -1:
            self.left_child[p] = rsib
        else:
            self.right_sib[lsib] = rsib
        if rsib == -1:
            self.right_child[p] = lsib
        else:
            self.left_sib[rsib] = lsib
        self.parent[c] = -1
        self.left_sib[c] = -1
        self.right_sib[c] = -1

    def insert_branch(self, p, c):
        self.parent[c] = p
        u = self.right_child[p]
        if u == -1:
            self.left_child[p] = c
            self.left_sib[c] = -1
            self.right_sib[c] = -1
        else:
            self.right_sib[u] = c
            self.left_sib[c] = u
            self.right_sib[c] = -1
        self.right_child[p] = c

    def remove_root(self, root):
        self.remove_branch(self.virtual_root, root)

    def insert_root(self, root):
        self.insert_branch(self.virtual_root, root)
        self.parent[root] = -1

    def remove_edge(self, p, c):
        assert p != -1
        self.remove_branch(p, c)
        # check for root changes
        u = p
        while u != tskit.NULL:
            path_end = u
            path_end_was_root = self.num_samples[u] > 0
            self.num_samples[u] -= self.num_samples[c]
            u = self.parent[u]
        if path_end_was_root and (self.num_samples[path_end] == 0):
            self.remove_root(path_end)
        if self.num_samples[c] > 0:
            self.insert_root(c)

    def insert_edge(self, p, c):
        assert p != -1
        assert self.parent[c] == -1, "contradictory edges"
        # check for root changes
        u = p
        while u != tskit.NULL:
            path_end = u
            path_end_was_root = self.num_samples[u] > 0
            self.num_samples[u] += self.num_samples[c]
            u = self.parent[u]
        if self.num_samples[c] > 0:
            self.remove_root(c)
        if (self.num_samples[path_end] > 0) and not path_end_was_root:
            self.insert_root(path_end)
        self.insert_branch(p, c)

    def push_down(self, n):
        """
        Push down references in the stack from n to other nodes
        to the children of n.
        """
        z = self.out[n]
        if n == self.focal_node:
            # HERE'S WHERE THE INPUT COMES IN
            if self.verbosity > 1:
                print(f"At {self.position}, updating since {self.x[n]} (for {n}).")
            z += self.position - self.x[n]
        c = self.left_child[n]
        while c != tskit.NULL:
            if self.verbosity > 1:
                print(f"adding {z} to {c}")
            self.out[c] += z
            c = self.right_sib[c]
        self.out[n] = 0
        self.x[n] = self.position

    def clear_spine(self, n):
        """
        Clears all nodes on the path from the virtual root down to n
        by pushing the contributions of all nodes to their children.
        """
        spine = []
        p = n
        while p != tskit.NULL:
            spine.append(p)
            p = self.parent[p]
        spine.append(self.virtual_root)
        for j in range(len(spine) - 1, -1, -1):
            p = spine[j]
            self.push_down(p)

    def run(self, node):
        self.focal_node = node
        sequence_length = self.sequence_length
        M = self.edges_left.shape[0]
        in_order = self.edge_insertion_order
        out_order = self.edge_removal_order
        edges_left = self.edges_left
        edges_right = self.edges_right
        edges_parent = self.edges_parent
        edges_child = self.edges_child

        j = 0
        k = 0
        # TODO: self.position is redundant with left
        self.position = left = 0

        while k < M and left <= self.sequence_length:
            while k < M and edges_right[out_order[k]] == left:
                p = edges_parent[out_order[k]]
                c = edges_child[out_order[k]]
                self.clear_spine(p)
                assert self.parent[p] == tskit.NULL or self.x[p] == self.position
                self.remove_edge(p, c)
                k += 1
                if self.verbosity > 1:
                    self.print_state(f"remove {p, c}")
            while j < M and edges_left[in_order[j]] == left:
                p = edges_parent[in_order[j]]
                c = edges_child[in_order[j]]
                if self.position > 0:
                    self.clear_spine(p)
                assert self.parent[p] == tskit.NULL or self.x[p] == self.position
                self.insert_edge(p, c)
                j += 1
                if self.verbosity > 1:
                    self.print_state(f"add {p, c}")
            right = sequence_length
            if j < M:
                right = min(right, edges_left[in_order[j]])
            if k < M:
                right = min(right, edges_right[out_order[k]])
            self.position = left = right
        assert j == M and left == self.sequence_length
        # clear remaining things down to virtual samples
        for u in self.samples:
            self.push_down(u)
        out = self.out[self.virtual_root + 1 + np.arange(len(self.samples))]
        return out


def inherits_from(ts, node, **kwargs):
    dm = InheritsFrom(
        ts.num_nodes,
        samples=ts.samples(),
        nodes_time=ts.nodes_time,
        edges_left=ts.edges_left,
        edges_right=ts.edges_right,
        edges_parent=ts.edges_parent,
        edges_child=ts.edges_child,
        edge_insertion_order=ts.indexes_edge_insertion_order,
        edge_removal_order=ts.indexes_edge_removal_order,
        sequence_length=ts.sequence_length,
        **kwargs
    )
    return dm.run(node)

def naive_inherits_from(ts, node):
    out = np.zeros(ts.num_samples, dtype='float64')
    for t in ts.trees():
        span = t.interval[1] - t.interval[0]
        for j, n in enumerate(ts.samples()):
            p = n
            while p != tskit.NULL:
                if p == node:
                    out[j] += span
                p = t.parent(p)
    return out

def small_example():
    ts = msprime.sim_ancestry(
        4,
        ploidy=1,
        population_size=10,
        sequence_length=10,
        recombination_rate=0.01,
        random_seed=124,
    )
    for t in ts.trees():
        print(t.interval)
        print(t.draw(format='ascii'))
    node = 0
    D1 = naive_inherits_from(ts, node=node)
    D2 = inherits_from(ts, node=node, verbosity=2)
    print(f"========trees: {ts.num_trees}=============")
    print("new:", D1)
    print("naive:", D2)
    print("=====================")
    assert np.allclose(D1, D2)


def verify():
    for seed in range(1, 10):
        ts = msprime.sim_ancestry(
            10,
            ploidy=1,
            population_size=20,
            sequence_length=100,
            recombination_rate=0.01,
            random_seed=seed,
        )
        t = ts.tables
        n = t.nodes.add_row(time=-1, flags=tskit.NODE_IS_SAMPLE)
        t.edges.add_row(parent=0, child=n, left=0, right=ts.sequence_length)
        t.sort()
        t.build_index()
        ts = t.tree_sequence()
        print(f"========{ts.num_trees}=============")
        for node in range(ts.num_nodes):
            D1 = naive_inherits_from(ts, node=node)
            D2 = inherits_from(ts, node=node)
            if not np.allclose(D1, D2):
                print("naive:", D1)
                print("  new:", D2)
            assert np.allclose(D1, D2)
        for x, y in zip(D1, D2):
            print(x, y)
        print("=====================")


if __name__ == "__main__":

    np.set_printoptions(linewidth=500, precision=4)
    small_example()
    verify()
