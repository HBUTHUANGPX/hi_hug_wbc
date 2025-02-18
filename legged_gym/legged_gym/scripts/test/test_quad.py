import torch

def is_point_in_quadrilateral(quadrilateral_point: torch.Tensor,
                                root_states: torch.Tensor) -> torch.Tensor:
    """
    判断每个环境中的root_states是否在对应的四边形内部。
    
    参数：
        quadrilateral_point: Tensor，形状 [num_envs, 4, 2]，
                             存放每个环境中四边形的四个顶点，
                             点的顺序依次为左上、左下、右下、右上。
        root_states: Tensor，形状 [num_envs, 2]，每个环境中待检测的点。
    
    返回：
        inside: Bool型Tensor，形状 [num_envs]，每个元素表示对应环境中
                root_states 点是否在四边形内部（True表示在内部，False表示不在内部）。
    """
    # 将四边形各个顶点沿第1维滚动（即对每个环境，将每个点的“下一个”顶点计算出来）
    quadr_next = torch.roll(quadrilateral_point, shifts=-1, dims=1)  # 形状 [num_envs, 4, 2]
    
    # 计算每条边的向量：edge = next_vertex - current_vertex
    edge_vectors = quadr_next - quadrilateral_point  # 形状 [num_envs, 4, 2]
    
    # 计算从每个顶点到待判断点的向量：相对向量
    # 注意：root_states的形状 [num_envs,2]扩展成 [num_envs,1,2]，与四边形顶点相减
    rel_vectors = root_states.unsqueeze(1) - quadrilateral_point  # 形状 [num_envs, 4, 2]
    
    # 计算二维叉积：对于二维向量 (a, b) 和 (c, d)，叉积的标量为 a*d - b*c
    cross_products = edge_vectors[..., 0] * rel_vectors[..., 1] - edge_vectors[..., 1] * rel_vectors[..., 0]  # 形状 [num_envs, 4]
    
    # 判断：如果在某个环境中，所有边的叉积均大于等于0或均小于等于0，则点在内部
    inside = torch.all(cross_products >= 0, dim=1) | torch.all(cross_products <= 0, dim=1)
    return inside

# 示例用法：
if __name__ == '__main__':
    # 假设设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 假设有 num_envs 个环境
    num_envs = 4
    
    # 构造示例：3个环境，每个环境的四边形顶点 (左上、左下、右下、右上)
    quadrilateral_point = torch.tensor([
        [[0.0, 1.0], [0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],  # 正方形
        [[-1.0, 2.0], [-1.0, 0.0], [2.0, 0.0], [2.0, 2.0]],  # 大一点的正方形
        [[0.0, 0.0], [0.0, -1.0], [1.0, -1.0], [1.0, 0.0]],   # 单位正方形（下移）
        [[0.0, 0.0], [0.0, -1.0], [1.0, -1.0], [1.0, 0.0]]   # 单位正方形（下移）
    ], dtype=torch.float, device=device)
    print(quadrilateral_point.size())
    # 构造示例：每个环境的检测点
    root_states = torch.tensor([
        [0.5, 0.5],   # 在第1个正方形内部
        [0.0, 1.0],   # 在边上（可认为内部或外部，根据具体需求；此处返回True，因为所有叉积可能为0）
        [1.5, 0.5],    # 在第3个正方形外部
        [1, 0]    # 在第3个正方形外部
    ], dtype=torch.float, device=device)
    print(root_states.size())
    
    # 计算布尔结果
    inside = is_point_in_quadrilateral(quadrilateral_point, root_states)
    print("检测点是否在四边形内部：", inside)  # 预期输出类似 [True, True, False]
