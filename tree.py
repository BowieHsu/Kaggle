#encoding = utf-8
__author__ = 'xubowen'

import numpy as np
import math

class Tree():
    def __init__(self, col = -1, split_feature = None, results = None, left_tree = None, right_tree = None):
        self.split_feature  = split_feature
        self.left_tree      = left_tree
        self.right_tree     = right_tree
        self.col            = col
        self.results        = results


    def SplitData(self, data, column, value):
        '''
        :param data:
        :param value:
        :return:
        '''
        valid_data = np.array(data)
        split_function = lambda row: row[column - 1] >= value

        set_1 = [row for row in data if split_function(row)]
        set_2 = [row for row in data if not split_function(row)]

        return (set_1, set_2)

    #iterate all the elements in the data
    def UniqueCounts(self, data):
        count = {}
        for row in data:
            r = row[len(data[0])- 1]
            if r not in count: count[r] = 0
            count[r] += 1
        return count

    #entropy calculate
    def Entropy(self, data):
        '''
        :param data:
        :return:
        '''

        log_value = lambda x:math.log(x)/math.log(2)
        count = self.UniqueCounts(data)
        data_len = len(data)
        ent = 0.0

        for i in count.keys():
            p = 1.0 * count[i]/data_len
            ent = ent - p * log_value(p)
        return ent


    def BuildTreeSubProcess(self, data):
        '''
        :param data:
        :return:
        '''

        (row_len, col_len) = data.shape
        col_len -= 1

        score_cur = self.Entropy(data)
        best_gain = 0.0
        best_set = None
        best_criteria = None

        for col in range(col_len):
            count = {}
            for row in data:
                count[row[col]] = 1
            for i in count.keys():
                (set_1, set_2) = self.SplitData(data, col, count[i])
                valid_p = float(len(set_1))/ row_len
                gain =  score_cur - valid_p * self.Entropy(set_1) -(1.0 - valid_p) * self.Entropy(set_2)
                if gain > best_gain and len(set_1) and len(set_2):
                    best_gain = gain
                    best_set  = (set_1, set_2)
                    best_criteria = (col, i)

        if best_gain <= 0.0 :
            return Tree(results = self.UniqueCounts(data))
        else:
            left_tree = self.BuildTreeSubProcess(best_set[0])
            right_tree = self.BuildTreeSubProcess(best_set[1])
            return Tree(left_tree = left_tree, right_tree = right_tree, col= best_criteria[0], split_feature= best_criteria[1])


    def BuildTree(self, data, label):
        '''
        :param data:
        :param label:
        :return:
        '''
        valid_data = np.array(data)
        valid_label = np.array(label)
        data_with_label = np.hstack((valid_data,valid_label))
        print data_with_label.shape

        return self.BuildTreeSubProcess(data_with_label)

    def Prune(self, tree, min_gain):
        '''
        :param tree:
        :param min_gain:
        :return:
        '''
        gain = 0.0
        if tree.left_tree == None or tree.right_tree == None:
            return
        if tree.left_tree.results == None:
            self.Prune(tree.left_tree, min_gain)
        elif tree.right_tree.results == None:
            self.Prune(tree.right_tree, min_gain)
        else:
            right_tree = []
            left_tree  = []

            for keys,values in tree.left_tree.results.items:
                left_tree += [[keys]] * values

            for keys,values in tree.right_tree.results.items:
                right_tree += [[keys]] * values

            gain = self.Entropy(left_tree + right_tree) - (self.Entropy(left_tree) + self.Entropy(right_tree)) * 0.5

            if gain < min_gain:
                tree.left_tree, tree.right_tree = None, None
                tree.results = self.UniqueCounts(left_tree + right_tree)



if __name__ == '__main__':

    train_data = [[1,2,3,4],[1,2,7,4]]

    label = [[1],[2]]

    cart = Tree()
    (set_1, set_2) = cart.SplitData(train_data, 3, 4)
    print set_1
    print set_2

    count = cart.UniqueCounts(train_data)

    tree = cart.BuildTree(train_data, label)

    prune_tree = cart.Prune(tree, 0.0)
