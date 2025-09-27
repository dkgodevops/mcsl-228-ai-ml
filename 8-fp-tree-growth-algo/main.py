"""
FP-Growth (Frequent Pattern Growth) — from-scratch Python implementation

- Build an FP-tree from a list of transactions
- Mine the tree to extract frequent itemsets
- Demo dataset: small synthetic retail transactions

Requirements: only Python standard library + typing (built-in).
"""

from collections import defaultdict, Counter
from typing import Dict, List, Optional, Tuple, Iterable, Any

class FPNode:
    def __init__(self, item: Optional[Any], count: int, parent: Optional["FPNode"]):
        self.item = item
        self.count = count
        self.parent = parent
        self.children: Dict[Any, "FPNode"] = {}
        self.link: Optional["FPNode"] = None  # next node with same item (header table)

    def increment(self, n: int = 1):
        self.count += n

    def display(self, ind=1):
        print('  ' * ind, f'{self.item}:{self.count}')
        for child in self.children.values():
            child.display(ind + 1)

class FPTree:
    def __init__(self, transactions: Iterable[Iterable[Any]], min_support: int):
        self.root = FPNode(None, 1, None)
        self.header_table: Dict[Any, Tuple[int, Optional[FPNode]]] = {}  # item -> (support, head_node)
        self.min_support = min_support
        self.build_tree(transactions)

    def build_tree(self, transactions: Iterable[Iterable[Any]]):
        # First pass: count frequency
        freq: Counter = Counter()
        for trans in transactions:
            freq.update(trans)
        # Remove infrequent items and store support in header_table
        self.header_table = {item: (count, None) for item, count in freq.items() if count >= self.min_support}
        if not self.header_table:
            return

        # Order items by descending support for insertion order
        def sort_transaction(tx):
            items = [i for i in tx if i in self.header_table]
            # sort by support desc, then by item name to have deterministic order
            items.sort(key=lambda i: (-self.header_table[i][0], str(i)))
            return items

        # Second pass: insert transactions
        for trans in transactions:
            ordered = sort_transaction(trans)
            if ordered:
                self._insert_tree(ordered, self.root)

    def _insert_tree(self, items: List[Any], node: FPNode):
        first = items[0]
        child = node.children.get(first)
        if child:
            child.increment(1)
        else:
            child = FPNode(first, 1, node)
            node.children[first] = child
            # update header table link
            count, head = self.header_table[first]
            if head is None:
                self.header_table[first] = (count, child)
            else:
                # follow link chain to append
                current = head
                while current.link is not None:
                    current = current.link
                current.link = child
        remaining = items[1:]
        if remaining:
            self._insert_tree(remaining, child)

    def conditional_pattern_base(self, item: Any) -> List[Tuple[List[Any], int]]:
        """
        For a given item, traverse nodes via header links and
        collect the prefix paths (excluding the item node) with their counts.
        Returns list of (prefix_path_items, count)
        """
        base: List[Tuple[List[Any], int]] = []
        _, node = self.header_table[item]
        while node is not None:
            count = node.count
            path = []
            parent = node.parent
            while parent is not None and parent.item is not None:
                path.append(parent.item)
                parent = parent.parent
            if path:
                base.append((list(reversed(path)), count))  # path should be from root->leaf order
            node = node.link
        return base

    def mine_patterns(self) -> Dict[Tuple[Any, ...], int]:
        """Mine the FP-tree and return frequent patterns with supports as a dict."""
        patterns: Dict[Tuple[Any, ...], int] = {}
        items = sorted(self.header_table.items(), key=lambda x: x[1][0])  # sort by increasing support (asc)
        for item, (support, _) in items:
            # singleton pattern
            patterns[(item,)] = support
            # build conditional pattern base and conditional tree
            base = self.conditional_pattern_base(item)
            # expand base into transactions repeated by count
            conditional_transactions = []
            for prefix, cnt in base:
                # each prefix path contributes 'cnt' identical transactions
                conditional_transactions.extend([prefix] * cnt)
            # build conditional FP-tree
            cond_tree = FPTree(conditional_transactions, self.min_support)
            # recursively mine conditional tree
            if cond_tree.header_table:
                cond_patterns = cond_tree.mine_patterns()
                for pat, sup in cond_patterns.items():
                    new_pat = tuple(sorted(tuple(pat) + (item,), key=str))
                    patterns[new_pat] = sup
        return patterns

# ---------------------- Demo ----------------------
if __name__ == "__main__":
    # Example dataset: small retail-like transactions
    transactions = [
        ['bread', 'milk'],
        ['bread', 'diaper', 'beer', 'egg'],
        ['milk', 'diaper', 'beer', 'cola'],
        ['bread', 'milk', 'diaper', 'beer'],
        ['bread', 'milk', 'diaper', 'cola'],
        ['cola', 'egg'],
        ['bread', 'egg'],
        ['milk', 'egg'],
        ['diaper', 'cola'],
        ['bread', 'milk', 'cola'],
    ]

    # convert to lists (already lists). You can replace with any list of transactions.
    min_support = 2  # absolute support (at least 2 transactions)
    tree = FPTree(transactions, min_support)

    print("=== FP-tree (display) ===")
    tree.root.display()

    print("\n=== Header table (item -> support) ===")
    for item, (sup, node) in sorted(tree.header_table.items(), key=lambda x: (-x[1][0], x[0])):
        print(f"{item}: {sup}")

    print("\n=== Conditional pattern base for 'diaper' ===")
    cpb = tree.conditional_pattern_base('diaper')
    for path, cnt in cpb:
        print(f"path={path}, count={cnt}")

    print("\n=== Mining frequent patterns (min_support = {}) ===".format(min_support))
    patterns = tree.mine_patterns()

    # Sort and print frequent patterns by support desc, length desc
    sorted_patterns = sorted(patterns.items(), key=lambda x: (-x[1], -len(x[0]), x[0]))
    for pat, sup in sorted_patterns:
        print(f"{list(pat)} -> support = {sup}")

    # If you prefer minimum support as fraction, convert:
    # min_support_fraction = 0.2  # 20%
    # absolute_min_support = int(min_support_fraction * len(transactions))
