from matplotlib import pyplot as plt

# 图表标题
font_title = {
    'weight': 'regular',
    'size': 12
}

# 坐标轴标题
font_label = {
    'weight': 'regular',
    'size': 10
}

# 图例
font_legend = {
    'weight': 'regular',
    'size': 6
}


def init_graph(xlabel, ylabel, title, dpi=150, style="seaborn-bright"):
    # 设置清晰度
    plt.figure(dpi=dpi)

    # 设置样式
    plt.style.use(style)

    # 添加x，y轴名称
    plt.xlabel(fontdict=font_label, xlabel=xlabel)
    plt.ylabel(fontdict=font_label, ylabel=ylabel)

    # 添加标题
    plt.title(title, font=font_title)