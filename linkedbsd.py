"""
File: linkedbst.py
Author: Ken Lambert
"""

from abstractcollection import AbstractCollection
from bstnode import BSTNode
from linkedstack import LinkedStack
# from linkedqueue import LinkedQueue
# from math import log
import timeit
import random


class LinkedBST(AbstractCollection):
    """An link-based binary search tree implementation."""

    def __init__(self, sourceCollection=None):
        """Sets the initial state of self, which includes the
        contents of sourceCollection, if it's present."""
        self._root = None
        AbstractCollection.__init__(self, sourceCollection)

    # Accessor methods
    def __str__(self):
        """Returns a string representation with the tree rotated
        90 degrees counterclockwise."""

        def recurse(node, level):
            sec = ""
            if node != None:
                sec += recurse(node.right, level + 1)
                sec += "| " * level
                sec += str(node.data) + "\n"
                sec += recurse(node.left, level + 1)
            return sec

        return recurse(self._root, 0)

    def show_root(self):
        """Method to show root"""
        return self._root

    def __iter__(self):
        """Supports a preorder traversal on a view of self."""
        if not self.isEmpty():
            stack = LinkedStack()
            stack.push(self._root)
            while not stack.isEmpty():
                node = stack.pop()
                yield node.data
                if node.right != None:
                    stack.push(node.right)
                if node.left != None:
                    stack.push(node.left)

    def preorder(self):
        """Supports a preorder traversal on a view of self."""
        return None

    def inorder(self):
        """Supports an inorder traversal on a view of self."""
        lyst = list()

        def recurse(node):
            if node != None:
                recurse(node.left)
                lyst.append(node.data)
                recurse(node.right)

        recurse(self._root)
        return iter(lyst)

    def postorder(self):
        """Supports a postorder traversal on a view of self."""
        return None

    def levelorder(self):
        """Supports a levelorder traversal on a view of self."""
        return None

    def __contains__(self, item):
        """Returns True if target is found or False otherwise."""
        return self.find(item) != None

    def find(self, item):
        """If item matches an item in self, returns the
        matched item, or None otherwise."""

        def recurse(node):
            if node is None:
                return None
            elif item == node.data:
                return node.data
            elif item < node.data:
                return recurse(node.left)
            else:
                return recurse(node.right)

        return recurse(self._root)

    # Mutator methods
    def clear(self):
        """Makes self become empty."""
        self._root = None
        self._size = 0

    def add(self, item):
        """Adds item to the tree."""

        # Helper function to search for item's position
        def recurse(node):
            # New item is less, go left until spot is found
            if item < node.data:
                if node.left == None:
                    node.left = BSTNode(item)
                else:
                    recurse(node.left)
            # New item is greater or equal,
            # go right until spot is found
            elif node.right == None:
                node.right = BSTNode(item)
            else:
                recurse(node.right)
                # End of recurse

        # Tree is empty, so new item goes at the root
        if self.isEmpty():
            self._root = BSTNode(item)
        # Otherwise, search for the item's spot
        else:
            recurse(self._root)
        self._size += 1

    def remove(self, item):
        """Precondition: item is in self.
        Raises: KeyError if item is not in self.
        postcondition: item is removed from self."""
        if not item in self:
            raise KeyError("Item not in tree.""")

        # Helper function to adjust placement of an item
        def lift_max_in_left_subtree_to_top(top):
            # Replace top's datum with the maximum datum in the left subtree
            # Pre:  top has a left child
            # Post: the maximum node in top's left subtree
            #       has been removed
            # Post: top.data = maximum value in top's left subtree
            parent = top
            current_node = top.left
            while not current_node.right == None:
                parent = current_node
                current_node = current_node.right
            top.data = current_node.data
            if parent == top:
                top.left = current_node.left
            else:
                parent.right = current_node.left

        # Begin main part of the method
        if self.isEmpty(): return None

        # Attempt to locate the node containing the item
        item_removed = None
        pre_root = BSTNode(None)
        pre_root.left = self._root
        parent = pre_root
        direction = 'L'
        current_node = self._root
        while not current_node == None:
            if current_node.data == item:
                item_removed = current_node.data
                break
            parent = current_node
            if current_node.data > item:
                direction = 'L'
                current_node = current_node.left
            else:
                direction = 'R'
                current_node = current_node.right

        # Return None if the item is absent
        if item_removed == None: return None

        # The item is present, so remove its node

        # Case 1: The node has a left and a right child
        #         Replace the node's value with the maximum value in the
        #         left subtree
        #         Delete the maximium node in the left subtree
        if not current_node.left == None \
                and not current_node.right == None:
            lift_max_in_left_subtree_to_top(current_node)
        else:

            # Case 2: The node has no left child
            if current_node.left == None:
                new_child = current_node.right

                # Case 3: The node has no right child
            else:
                new_child = current_node.left

                # Case 2 & 3: Tie the parent to the new child
            if direction == 'L':
                parent.left = new_child
            else:
                parent.right = new_child

        # All cases: Reset the root (if it hasn't changed no harm done)
        #            Decrement the collection's size counter
        #            Return the item
        self._size -= 1
        if self.isEmpty():
            self._root = None
        else:
            self._root = pre_root.left
        return item_removed

    def replace(self, item, new_item):
        """
        If item is in self, replaces it with newItem and
        returns the old item, or returns None otherwise."""
        probe = self._root
        while probe != None:
            if probe.data == item:
                old_data = probe.data
                probe.data = new_item
                return old_data
            elif probe.data > item:
                probe = probe.left
            else:
                probe = probe.right
        return None


    def height(self):
        """
        Returns the height of the tree.
        :return: The height of the tree as an integer.
        """

        return self.get_height(self._root)


    def get_height(self, node):
        """
        Height helper function
        """
        if node is None:
            return 0

        if not node.left and not node.right:
            return 0

        left_height = self.get_height(node.left)
        right_height = self.get_height(node.right)
        return max(left_height, right_height) + 1


    def get_balance(self, root_node):
        '''
        Return True if tree is balanced
        :return:
        '''
        if root_node is None:
            return True

        left_height = self.get_height(root_node.left)
        right_height = self.get_height(root_node.right)


        if (abs(left_height - right_height) <= 1 and
            self.get_balance(root_node.left) is True and
            self.get_balance(root_node.right) is True):
            return True

        return False


    def is_balanced(self):
        '''
        Return True if tree is balanced
        :return:
        '''
        return self.get_balance(self._root)


    def range_find(self, low, high):
        '''
        Returns a list of the items in the tree, where low <= item <= high."""
        :param low:
        :param high:
        :return:
        '''
        items = []

        def walk(node):
            """walk"""
            if node is None:
                return None

            walk(node.right)
            walk(node.left)

            if low <= node.data <= high:
                items.append(node.data)

        walk(self._root)
        return sorted(items)

    def rebalance(self):
        '''
        Rebalances the tree.
        :return:
        '''
        sorted_items = list(self.inorder())
        self.clear()
        self._root = self._build_balanced_tree(sorted_items)

    def _build_balanced_tree(self, items):
        """Recursively builds a balanced binary search tree from a sorted list of items."""
        if not items:
            return None

        mid = len(items) // 2
        root = BSTNode(items[mid])

        root.left = self._build_balanced_tree(items[:mid])
        root.right = self._build_balanced_tree(items[mid+1:])

        return root

    def successor(self, item):
        """
        Returns the smallest item that is larger than
        item, or None if there is no such item.
        :param item:
        :type item:
        :return:
        :rtype:
        """
        current = self._root
        successor = None

        while current is not None:
            if current.data > item:
                successor = current.data
                current = current.left
            else:
                current = current.right

        return successor

    def predecessor(self, item):
        """
        Returns the largest item that is smaller than
        item, or None if there is no such item.
        :param item:
        :type item:
        :return:
        :rtype:
        """
        current = self._root
        successor = None

        while current is not None:
            if current.data < item:
                successor = current.data
                current = current.right
            else:
                current = current.left

        return successor

    def read_file(self, path):
        """
        Method to read data from file
        """
        with open(path, 'r', encoding="utf-8") as file:
            data = file.read().strip().split('\n')

        return data

    def linear_search(self, data, data_to_search):
        """
        Linear_search method to search num random elements
        """
        indexes = []
        for index, elem in enumerate(data):
            if elem in data_to_search:
                indexes.append((elem, index))

    def walk_tree(self, root):
        """Tree traversal"""
        items = []

        def walk(node, ind=0, side = ''):
            """walk"""
            if node is None:
                return None

            walk(node.right, ind + 1, 'r')
            walk(node.left, ind + 1, 'r')
            items.append((node.data, ind, side))

        walk(root)
        return items

    def demo_bst(self, path):
        """
        Demonstration of efficiency binary search tree for the search tasks.
        :param path:
        :type path:
        :return:
        :rtype:
        """
        num = 1000
        data = self.read_file(path)
        data_to_search = random.choices(data, k=num)

        # append words to tree
        sample_tree = LinkedBST()
        for elem in data:
            sample_tree.add(elem)

        # linear search resuts
        result1 = (timeit.timeit(stmt=lambda: self.linear_search(data, data_to_search),
                                 globals=globals()))

        # binary tree search
        result2 = (timeit.timeit(stmt=lambda: self.walk_tree(sample_tree.show_root()),
                                 globals=globals()))

        # append words to tree randomly
        sample_tree_2 = LinkedBST()
        shuffled_data = data.copy()
        random.shuffle(shuffled_data)
        for elem in shuffled_data:
            sample_tree_2.add(elem)

        # shuffled binary tree search
        result3 = (timeit.timeit(stmt=lambda: self.walk_tree(sample_tree_2.show_root()),
                                 globals=globals()))

        # rebalanced binary tree
        sample_tree_2.rebalance()
        result4 = (timeit.timeit(stmt=lambda: self.walk_tree(sample_tree_2.show_root()),
                                 globals=globals()))

        return (f"Linear search {result1} sec\n" +
               f"Binary tree search {result2} sec\n"+
               f"Shuffled binary tree search {result3} sec\n"+
               f"Rebalanced binary tree search {result4} sec\n")

if __name__ == "__main__":
    tree = LinkedBST()
    print(tree.demo_bst("words.txt"))
