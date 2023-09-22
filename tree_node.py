import numpy as np
from collections import deque

class TreeNode:
    def __init__(self, id,variable,isfinal,level):
        self.variable = variable
        self.isfinal = isfinal
        self.id = id
        self.children = []  # 子节点列表
        self.level = level

    def add_child(self, child_node):
        self.children.append(child_node)
    
    def __str__(self):
        return f"id: {self.id}, variable: {self.variable}, isfinal: {self.isfinal}"


# 创建根节点
root_value = input("请输入根节点的id: ")
root = TreeNode(root_value,0,False,1)


# 交互式构建树
while True:
    print(f"当前树的结构:")
    def print_tree(node, level = 0):
        if node:
            print("  " * level + f"id: {node.id} variable: {node.variable} is_final: {node.isfinal} level:{node.level}")
            for child in node.children:
                print_tree(child, node.level)

    print_tree(root)

    choice = input("选择要操作的节点 (输入节点值) 或输入 'q' 退出: ")

    if choice == 'q':
        break

    input_str= input(f"请输入要添加到节点 {choice} 的子节点的值: ")

    
    add_list = input_str.split(' ')

    while len(add_list) != 3:
        print("添加节点需要输入三个参数: id_to_add,variable_to_add,is_final_to_add")
        input_str= input(f"请输入要添加到节点 {choice} 的子节点的值: ")
        add_list = input_str.split(' ')

    id_to_add = add_list[0]
    variable_to_add = add_list[1]
    is_final_to_add = add_list[2]

    # 在树中查找要添加子节点的节点
    def find_node(node, id):
        if node.id == id:
            return node
        for child in node.children:
            found_node = find_node(child, id)
            if found_node:
                return found_node
        return None

    selected_node = find_node(root, choice)

    if selected_node:
        selected_node.add_child(TreeNode(id_to_add,variable_to_add,is_final_to_add,selected_node.level+1))
        print(f"节点 {id_to_add} 已添加到节点 {choice},variable = {variable_to_add},isfinal = {is_final_to_add},level = {selected_node.level+1}")
    else:
        print(f"节点 {choice} 未找到.")



# 计算奇数层和偶数层的边数
def count_edges_on_levels(root):
    if not root:
        return 0, 0, 0, 0  # 奇数层和偶数层的边数初始值为0

    odd_level_edges = 0
    odd_levels = 1
    even_level_edges = 0
    even_levels = 0

    queue = deque()
    queue.append(root)  # 元组包含节点和层数
    print("root.children = ",root.children)

    while queue:
        node = queue.popleft()
        if node.level % 2 == 0:
            even_level_edges += len(node.children)
            if node.isfinal != 'True' and node.level < queue[0].level:
                even_levels += 1
        else:
            odd_level_edges += len(node.children)
            if len(queue) > 0:
                if node.isfinal != 'True' and node.level < queue[0].level:
                    odd_levels += 1

        for child in node.children:
            queue.append(child)

    return odd_level_edges, even_level_edges,odd_levels,even_levels

odd_edges, even_edges,odd_levels,even_levels = count_edges_on_levels(root)

A = np.zeros((odd_edges,even_edges))
B = np.zeros((odd_edges,even_edges))
E = np.zeros((odd_levels,odd_edges))
F = np.zeros((even_levels,even_edges))
print("E.shape =",E.shape)
print("F.shape = ",F.shape)
e = np.zeros((odd_levels,1))
e[0] = 1
f = np.zeros((even_levels,1))
f[0] = 1
print("e.shape = ",e.shape)
print("f.shape = ",f.shape)



# 执行层序遍历并记录层数和子节点的id
def level_order_traversal_with_level(root):
    if not root:
        return
    player1_row_number = 0
    player1_col_number = 0
    player2_row_number = 0
    player2_col_number = 0
    queue = deque()
    queue.append(root)  # 元组包含节点和层数
    while queue:
        node = queue.popleft()
        child_number = [child.id for child in node.children]  # 子节点的id列表
        if node.level % 2 == 0 and node.isfinal != True:
            print("node.level % 2 == 0 and node.isfinal != True:",node.isfinal)
            if player2_row_number == 0:
                temp_node_list = list()
                temp_node = node
                player2_first_row_number = len(temp_node.children)
                temp_node_list.append(temp_node)
                while temp_node.level == 2:
                    temp_node = queue.popleft()
                    temp_node_list.append(temp_node)
                    player2_first_row_number += len(temp_node.children)
                    if len(queue) == 0 :
                        break
                for queue_out_node in reversed(temp_node_list):
                    queue.appendleft(queue_out_node)
                for i in range(player2_first_row_number):
                    F[player2_row_number,i] = 1
                player2_row_number += 1
            elif len(node.children) != 0:
                F[player2_row_number,player2_col_number] = -1
                total_children = 0
                for node_children in node.children:
                    total_children += len(node_children.children)
                for i in range(total_children):
                    F[player2_row_number,player2_col_number + 1 + i] = 1
                player2_col_number += 1
                player2_row_number += 1
        else:
            if player1_row_number == 0:
                player1_first_row_number = len(node.children)
                for i in range(player1_first_row_number):
                    E[player2_row_number,i] = 1
                player1_row_number += 1
            else:
                E[player1_row_number,player1_col_number] = -1
                total_children = 0
                for node_children in node.children:
                    total_children += len(node_children.children)
                for i in range(total_children):
                    F[player1_row_number,player2_col_number + 1 + i] = 1
                player1_col_number += 1
                player1_row_number += 1

        for child in node.children:
            queue.append(child)
    return A,B,E,F,e,f


print("层序遍历结果:")
level_order_traversal_with_level(root)

A,B,E,F,e,f = level_order_traversal_with_level(root)
print("E = ",E)
print("F = ",F)