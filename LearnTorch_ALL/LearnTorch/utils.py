import os
import subprocess
import numpy as np
from LearnTorch import Variable

# =============================================================================
# 计算图可视化函数
# =============================================================================
def _dot_var(v, verbose=False): # 变量Variable转化为dot语言的函数
    dot_var = '{} [label="{}", color=lightcoral, style=filled]\n'
    name = str(v.data) if v.name is None else (v.name+": " + str(v.data))
    if verbose and v.data is not None:
        if v.name is not None:
            name += ': '
        name += str(v.shape) + ' ' + str(v.dtype)
    return dot_var.format(id(v), name)
    # 这里采用python内置的id函数来得到对象特有的ID，再作为节点的id
    
def _dot_func_(f): # 函数Function转化为dot语言的函数
    dot_func = '{} [label ="{}", color=lightblue, style=filled, shape=box]\n'
    txt = dot_func.format(id(f), f.__class__.__name__)

    dot_edge = '{} -> {}\n' # 表示边的变量
    for x in f.inputs: # 全部输入节点指向函数节点
        txt += dot_edge.format(id(x), id(f))
    for y in f.outputs:# 函数节点指向全部输出节点
        txt += dot_edge.format(id(f), id(y())) #因为y是weakref，所以要用y()来引用
    return txt

# 获取计算图的dot语言版本
def get_dot_graph(output, verbose=True):
    # 逻辑参照Variable中的回溯算法
    # 因为只需要显示，顺序不重要，所以取消了generation部分
    txt = ''
    funcs = []
    seen_set = set()

    def add_func(f):
        # 无重复后添加算法
        if f not in seen_set:
            funcs.append(f)
            seen_set.add(f)

    add_func(output.creator)
    txt += _dot_var(output, verbose) # 添加最终输出的变量

    while funcs:
        func = funcs.pop()
        txt += _dot_func_(func)
        for x in func.inputs:
            txt += _dot_var(x, verbose) # 添加函数的输入变量

            if x.creator is not None:
                add_func(x.creator) # 添加进一步的函数
    return 'digraph g {\n' + txt + '}'

#计算图的可视化函数
def plot_dot_graph(output, verbose=True, to_file='graph.png', file_path='CGMap'):
    # 最终输出变量 是否详细显示 输出文件名 输出路径
    dot_graph = get_dot_graph(output, verbose)

    # 将dot数据保存至文件
    if not os.path.exists(file_path): # 如果不存在就创建该目录
        os.makedirs(file_path) # 递归创建多层目录
    graph_path = os.path.join(file_path, os.path.splitext(to_file)[0] +'.dot') # 这里和书上不一样，保存dot文件

    with open(graph_path, "w") as f:
        f.write(dot_graph)
    
    to_file_path = os.path.join(file_path, to_file) # png文件路径
    extension = os.path.splitext(to_file)[1][1:]# 扩展名(png、pdf等)
    cmd = 'dot {} -T {} -o {}'.format(graph_path, extension, to_file_path)# dot文件名 目标文件类型 目标文件名
    subprocess.run(cmd, shell = True) # 使用subprocess.run调用cmd命令

    # 如果是Jupyter Notebook ，还会直接在单元格展示（返回Jupyter Notebook Image object）
    try:
        from IPython import display
        return display.Image(filename=to_file)
    except:
        pass

# =============================================================================
# 梯度下降可视化函数
# =============================================================================