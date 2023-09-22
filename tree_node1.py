import numpy as np
from collections import deque

class TreeNode:
    def __init__(self, level, edgeIndex = -1):
        self.level = level # 树中层数，1开始
        self.edgeIndex = edgeIndex # 是否有父节点，有的话就是F/E列标，-1代表没有此列
        self.hasAdditionalRow = False # 是否要新增一行
        self.children = []  # 子节点列表
        self.payoff = [None,None]

    def add_child(self, child_node):
        self.children.append(child_node)
    
    def __str__(self):
        return f"level: {self.level}, edgeIndex: {self.edgeIndex}, hasAdditionalRow: {self.hasAdditionalRow}"

# 创建
root = TreeNode(1)
node1 = TreeNode(2)
node2 = TreeNode(2)
node3 = TreeNode(3)
node4 = TreeNode(3)
node5 = TreeNode(3)
node6 = TreeNode(3)
node7 = TreeNode(3)
node8 = TreeNode(3)
node9 = TreeNode(4)
node10 = TreeNode(4)
node11 = TreeNode(4)
node12 = TreeNode(4)
node13 = TreeNode(4)
node14 = TreeNode(4)
node15 = TreeNode(4)
node16 = TreeNode(4)
node17 = TreeNode(5)
node18 = TreeNode(5)
node19 = TreeNode(5)
node20 = TreeNode(5)
node21 = TreeNode(6)
node22 = TreeNode(6)
node23 = TreeNode(6)
node24 = TreeNode(6)
node25 = TreeNode(6)
node26 = TreeNode(7)
node27 = TreeNode(7)
node28 = TreeNode(8)
node29 = TreeNode(9)
node30 = TreeNode(9)

node28.add_child(node29)
node28.add_child(node30)

node26.add_child(node28)
node25.add_child(node27)
node23.add_child(node26)
node20.add_child(node25)
node19.add_child(node24)
node18.add_child(node23)

node17.add_child(node21)
node17.add_child(node22)

node16.add_child(node20)
node14.add_child(node19)
node9.add_child(node17)
node9.add_child(node18)

node8.add_child(node16)
node7.add_child(node15)
node6.add_child(node14)
node5.add_child(node13)
node4.add_child(node12)
node3.add_child(node9)
node3.add_child(node10)
node3.add_child(node11)

node2.add_child(node7)
node2.add_child(node8)

node1.add_child(node3)
node1.add_child(node4)
node1.add_child(node5)
node1.add_child(node6)

root.add_child(node1)
root.add_child(node2)

# payoff init

node21.payoff[0],node21.payoff[1] = 20,5
node22.payoff[0],node22.payoff[1] = 13,6
node29.payoff[0],node29.payoff[1] = 30,45
node30.payoff[0],node30.payoff[1] = 120,70
node11.payoff[0],node11.payoff[1] = 11,11
node12.payoff[0],node12.payoff[1] = 6,18
node13.payoff[0],node13.payoff[1] = 12,12
node24.payoff[0],node24.payoff[1] = 108,64
node15.payoff[0],node15.payoff[1] = 79,32
node10.payoff[0],node10.payoff[1] = 15,8
node27.payoff[0],node27.payoff[1] = 32,79

# 查询一个节点下两层是否有边，返回边数和对应需要的新增行数
def findAvailableEdge(node):
    edgeCount = 0
    newRowCount = 0
    for child in node.children:
        init = edgeCount
        for subChild in child.children:
            edgeCount += len(subChild.children)
        if edgeCount > init:
            newRowCount += 1
    return edgeCount, newRowCount

# 查询一个节点的父节点（奇偶对调）是否需要额外添加对应边的行
def hasAdditionalRow(node):
    for child in node.children:
        if len(child.children) > 0:
            return True
    return False

# 补全节点信息，返回E/F矩阵维数
def count_edges_on_levels(root):

    # E/F矩阵维数
    ERow = FRow = 1
    ECol = FCol = 0

    # 边总数
    oddEdgeIndex = 0
    evenEdgeIndex = 0

    queue = deque()
    queue.append(root)

    while queue:

        node = queue.popleft()

        # 1,2层单独处理
        if node.level == 1:

            # 补全数据
            node.edgeIndex = -1 # 无父节点
            node.hasAdditionalRow = hasAdditionalRow(node)

            availableEdges, newRowCount = findAvailableEdge(node)
            if availableEdges != 0:
                ERow += newRowCount

            ECol += len(node.children)
            
            for child in node.children:
                queue.append(child)
            continue
        
        if node.level == 2:

            # 补全数据
            node.edgeIndex = oddEdgeIndex
            node.hasAdditionalRow = hasAdditionalRow(node)
            oddEdgeIndex += 1

            availableEdges, newRowCount = findAvailableEdge(node)
            if availableEdges != 0:
                FRow += newRowCount

            FCol += len(node.children)

            for child in node.children:
                queue.append(child)
            continue

        # odd
        if node.level % 2 == 1:

            # 补全数据
            node.hasAdditionalRow = hasAdditionalRow(node)
            availableEdges, newRowCount = findAvailableEdge(node)
            if availableEdges != 0:
                ERow += newRowCount

            ECol += len(node.children)

            node.edgeIndex = evenEdgeIndex
            evenEdgeIndex += 1

            for child in node.children:
                queue.append(child)
            continue

        # even
        if node.level % 2 == 0:

            # 补全数据
            node.hasAdditionalRow = hasAdditionalRow(node)
            availableEdges, newRowCount = findAvailableEdge(node)
            # print(node.level, availableEdges)
            if availableEdges != 0:
                FRow += newRowCount
            
            FCol += len(node.children)

            node.edgeIndex = oddEdgeIndex
            oddEdgeIndex += 1

            for child in node.children:
                queue.append(child)
            continue

    return ERow, ECol, FRow, FCol

ERow, ECol, FRow, FCol = count_edges_on_levels(root)

A = np.zeros((ECol,FCol))
B = np.zeros((ECol,FCol))
E = np.zeros((ERow,ECol))
F = np.zeros((FRow,FCol))
print("E.shape =",E.shape)
print("F.shape = ",F.shape)
e = np.zeros((ERow,1))
e[0] = 1
f = np.zeros((FRow,1))
f[0] = 1
print("e.shape = ",e.shape)
print("f.shape = ",f.shape)
print()

# 执行层序遍历产生矩阵E/F/A/B
def GenerateMatrixEandF(root):

    queue = deque()
    queue.append(root)

    ERowIndex = FRowIndex = 0
    Payoff_Matrix_row_Index = Payoff_Matrix_col_Index = 0
    
    while queue:
        
        node = queue.popleft()

        # odd
        if node.level % 2 == 1:
            for child in node.children:
                queue.append(child)

                # 第一行标1
                if node.level == 1:
                    E[0, child.edgeIndex] = 1
                else:
                    for node_children in node.children:
                        if not node_children.children:
                            Payoff_Matrix_row_Index = node_children.edgeIndex
                            Payoff_Matrix_col_Index = node.edgeIndex
                            A[Payoff_Matrix_row_Index,Payoff_Matrix_col_Index] = node_children.payoff[0]
                            B[Payoff_Matrix_row_Index,Payoff_Matrix_col_Index] = node_children.payoff[1]
                # 如果有其他行
                if child.hasAdditionalRow:
                    ERowIndex += 1
                    E[ERowIndex, child.edgeIndex] = -1
                    for subchild in child.children:
                        for subsubChild in subchild.children:
                            E[ERowIndex, subsubChild.edgeIndex] = 1
        # even
        else:
            for child in node.children:
                queue.append(child)

                # 第一行标1
                if node.level == 2:
                    F[0, child.edgeIndex] = 1
                else:
                    for node_children in node.children:
                        if not node_children.children:
                            Payoff_Matrix_col_Index = node_children.edgeIndex
                            Payoff_Matrix_row_Index = node.edgeIndex
                            A[Payoff_Matrix_row_Index,Payoff_Matrix_col_Index] = node_children.payoff[0]
                            B[Payoff_Matrix_row_Index,Payoff_Matrix_col_Index] = node_children.payoff[1]
                # 如果有其他行
                if child.hasAdditionalRow:
                    FRowIndex += 1
                    F[FRowIndex, child.edgeIndex] = -1
                    for subchild in child.children:
                        for subsubChild in subchild.children:
                            F[FRowIndex, subsubChild.edgeIndex] = 1

        if not node.children:
            A[Payoff_Matrix_row_Index,Payoff_Matrix_col_Index] = node.payoff[0]
            B[Payoff_Matrix_row_Index,Payoff_Matrix_col_Index] = node.payoff[1]
    return

GenerateMatrixEandF(root)
print("E = ",E)
print()
print("F = ",F)
print()
print("A = ")
print(A)
print()
print("B = ")
print(B)
