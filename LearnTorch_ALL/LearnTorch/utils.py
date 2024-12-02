import os
import subprocess
import numpy as np
from LearnTorch import Variable

# =============================================================================
# 计算图可视化函数
# =============================================================================
def _dot_var(v, direction, verbose=False): # 变量Variable转化为dot语言的函数
    if direction == "Forward": # 这里不能用is，而是要用==，因为不在乎内存地址是否一样
        dot_var = '{} [label="{}", color=lightcoral, style=filled]\n'
        name = "(" + direction + ")" + "data:" + str(v.data) if v.name is None else (v.name + "(" + direction + ")" + " data: " + str(v.data))
    elif direction == "Backward":
        dot_var = '{} [label="{}", color=lightgoldenrodyellow, style=filled]\n'
        name = "(" + direction + ")" + "data:" +str(v.data) + "grad:"+ str(v.grad) if v.name is None else (
                    v.name + "(" + direction + ")" + " data:" +str(v.data) + " grad:"+ str(v.grad))
    if verbose and v.data is not None:
        if v.name is not None:
            name += ': '
        name += str(v.shape) + ' ' + str(v.dtype)
    return dot_var.format(direction+str(id(v)), name)
    # 这里采用python内置的id函数来得到对象特有的ID，再作为节点的id
    
def _dot_func_(f, direction): # 函数Function转化为dot语言的函数
    dot_func = '{} [label ="{}", color=lightblue, style=filled, shape=box]\n'
    txt = dot_func.format(direction+str(id(f)), f.__class__.__name__)

    dot_edge = '{} -> {}\n' # 表示边的变量
    if direction == "Forward": # 如果要生成正向传播的图
        for x in f.inputs: # 全部输入节点指向函数节点
            txt += dot_edge.format(direction+str(id(x)), direction+str(id(f)))
        for y in f.outputs:# 函数节点指向全部输出节点
            txt += dot_edge.format(direction+str(id(f)), direction+str(id(y()))) #因为y是weakref，所以要用y()来引用
    elif direction == "Backward": # 如果要生成反向传播的图
        for x in f.inputs: # 函数节点指向所有输入节点
            txt += dot_edge.format(direction+str(id(f)), direction+str(id(x)))
        for y in f.outputs:# 全部输出节点指向函数节点
            txt += dot_edge.format(direction+str(id(y())), direction+str(id(f))) #因为y是weakref，所以要用y()来引用
    return txt

# 获取计算图的dot语言版本
def get_dot_graph(output, direction, verbose=True):
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
    txt += _dot_var(output, direction, verbose) # 添加最终输出的变量

    while funcs:
        func = funcs.pop()
        txt += _dot_func_(func, direction)
        for x in func.inputs:
            txt += _dot_var(x, direction,verbose) # 添加函数的输入变量

            if x.creator is not None:
                add_func(x.creator) # 添加进一步的函数
    return  txt

#计算图的可视化函数
def plot_dot_graph(output, save_file, verbose=False, to_file='graph.png', file_path='CGMap'):
    # 输入：最终输出变量 是否保存dot以外文件 是否详细显示 输出文件名 输出路径
    # 输出：dot语言代码
    dot_forward = get_dot_graph(output=output , direction = "Forward", verbose=verbose)
    dot_backward = get_dot_graph(output=output , direction = "Backward", verbose=verbose)
    dot_graph_forward = 'digraph g {\n' + dot_forward + '}'
    dot_graph_backward = 'digraph g {\n' + dot_backward + '}'
    dot_graph_all = (
        'digraph g {\n'
        'subgraph cluster_g1 {\n'
        '    label = "Forward Propagation";\n'
        '    color = blue;\n'
        '    style = dashed;\n'
        f'{dot_forward}\n'
        '}\n'
        'subgraph cluster_g2 {\n'
        '    label = "Backward Propagation";\n'
        '    color = red;\n'
        '    style = dashed;\n'
        '    rankdir=BT;\n'
        f'{dot_backward}\n'
        '}\n'
        '}\n'
    )



    # 将dot数据保存至文件
    if not os.path.exists(file_path): # 如果不存在就创建该目录
        os.makedirs(file_path) # 递归创建多层目录

    graph_forward_dot_path = os.path.join(file_path, os.path.splitext(to_file)[0] + "Forward" + '.dot') # 这里和书上不一样，保存前向传播计算图dot文件
    graph_backward_dot_path = os.path.join(file_path, os.path.splitext(to_file)[0] + "Backward" + '.dot')  # 这里和书上不一样，保存后向传播计算图dot文件
    graph_all_dot_path = os.path.join(file_path, os.path.splitext(to_file)[0] + "All" + '.dot')

    with open(graph_forward_dot_path, "w") as f: # 保存前向传播计算图dot文件
        f.write(dot_graph_forward)

    with open(graph_backward_dot_path, "w") as f: # 保存后向传播计算图dot文件
        f.write(dot_graph_backward)

    with open(graph_all_dot_path, "w") as f: # 保存共同计算图dot文件
        f.write(dot_graph_all)

    if save_file:
        to_file_forward_path = os.path.join(file_path, os.path.splitext(to_file)[0] + "_Forward" + os.path.splitext(to_file)[1]) # 前向传播目标文件路径
        to_file_backward_path = os.path.join(file_path, os.path.splitext(to_file)[0] + "_Backward" + os.path.splitext(to_file)[1])  # 后向传播目标文件路径
        to_file_all_path = os.path.join(file_path, os.path.splitext(to_file)[0] + "_All" + os.path.splitext(to_file)[1])  # 总计算图目标文件路径
        extension = os.path.splitext(to_file)[1][1:]# 扩展名(png、pdf等)
        cmd_forward = 'dot {} -T {} -o {}'.format(graph_forward_dot_path, extension, to_file_forward_path)# 前向传播dot文件名 目标文件类型 目标文件名
        cmd_backward = 'dot {} -T {} -o {}'.format(graph_backward_dot_path, extension, to_file_backward_path)# 后向传播dot文件名 目标文件类型 目标文件名
        cmd_all = 'dot {} -T {} -o {}'.format(graph_all_dot_path, extension, to_file_all_path)  # 总计算图dot文件名 目标文件类型 目标文件名
        subprocess.run(cmd_forward, shell = True) # 使用subprocess.run调用前向传播cmd命令
        subprocess.run(cmd_backward, shell = True)  # 使用subprocess.run调用后向传播cmd命令
        subprocess.run(cmd_all, shell = True)         # 使用subprocess.run调用总计算图cmd命令

        # 如果是Jupyter Notebook ，还会直接在单元格展示（返回Jupyter Notebook Image object）
        try:
            from IPython import display
            return display.Image(to_file_all_path)
        except:
            pass

# =============================================================================
# 梯度下降可视化函数
# =============================================================================