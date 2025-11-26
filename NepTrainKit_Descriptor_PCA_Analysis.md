# NepTrainKit Descriptor PCA 分析流程详解

## 概述

本文档详细分析了 NepTrainKit 库中从原始结构数据到 PCA 可视化的完整流程，特别关注如何处理不同原子数的结构，以及 descriptor 的聚合方法。

---

## 完整数据流程

```
原始结构 (xyz 文件)
    ↓
NEP 模型计算
    ↓
Per-atom descriptor (每个原子固定维度，如 35 维)
    ↓
聚合操作 (对每个结构的原子取平均)
    ↓
Per-structure descriptor (每个结构固定维度，如 35 维)
    ↓
PCA 降维
    ↓
2D 坐标点 (可视化)
```

---

## 关键步骤详解

### 1. NEP 模型计算 Per-atom Descriptor

**位置**: `.venv/lib/python3.13/site-packages/NepTrainKit/core/calculator.py`

```python
def get_structures_descriptor(self, structures: list[Structure], mean_descriptor: bool=True):
    # NEP 模型为每个原子计算 descriptor
    descriptor = self.nep3.get_structures_descriptor(types, boxes, positions)
    # descriptor.shape = (总原子数, descriptor_dim)
```

**特点**:
- NEP 为**每个原子**计算一个固定维度的 descriptor（通常 35 维）
- 无论结构有多少原子，每个原子的 descriptor 维度都相同
- 输出形状: `(总原子数, 35)`

**示例**:
```
100 个结构，每个 20 个原子
→ 2000 个原子 × 35 维 = (2000, 35) 矩阵
```

---

### 2. 聚合为 Per-structure Descriptor ⭐

**核心函数**: `aggregate_per_atom_to_structure`

**位置**: `.venv/lib/python3.13/site-packages/NepTrainKit/core/utils.py` (第 65-75 行)

```python
def aggregate_per_atom_to_structure(
    array: npt.NDArray[np.float32],      # Per-atom descriptor
    atoms_num_list: Iterable[int],       # 每个结构的原子数
    map_func=np.linalg.norm,             # 聚合函数（默认范数）
    axis: int = 0,                       # 聚合轴
) -> npt.NDArray[np.float32]:           # Per-structure descriptor
    """Aggregate per-atom data into per-structure values based on atom counts."""
    split_arrays = split_by_natoms(array, atoms_num_list)
    func = partial(map_func, axis=axis)
    return np.array(list(map(func, split_arrays)))
```

#### 函数工作原理

**步骤 1: 按原子数分割**
```python
split_arrays = split_by_natoms(array, atoms_num_list)
```

将大数组按每个结构的原子数切分：
```python
# 输入: (2000, 35)，原子数列表 [20, 20, 20, ...]
# 输出: [
#   array(20, 35),  # 结构 1
#   array(20, 35),  # 结构 2
#   ...
# ]
```

**步骤 2: 对每个结构应用聚合函数**
```python
func = partial(map_func, axis=0)
return np.array(list(map(func, split_arrays)))
```

对每个结构的所有原子应用聚合函数（默认 `np.mean`）：
```python
# 对每个结构: (20, 35) → np.mean(axis=0) → (35,)
# 最终输出: (100, 35)
```

#### 实际调用位置

**1. calculator.py (第 326 行)**
```python
structure_descriptor = aggregate_per_atom_to_structure(
    descriptor, 
    group_sizes, 
    map_func=np.mean,  # ✅ 使用平均值
    axis=0
)
```

**2. base.py (第 976 行)**
```python
desc_array = aggregate_per_atom_to_structure(
    desc_array, 
    self.atoms_num_list, 
    map_func=np.mean,  # ✅ 使用平均值
    axis=0
)
```

**3. sampler.py (第 258 行)**
```python
t_desc = aggregate_per_atom_to_structure(
    t_desc, 
    t_counts, 
    map_func=np.mean,  # ✅ 使用平均值
    axis=0
)
```

### ✅ 确认：使用简单平均化

**所有 descriptor 相关的调用都使用 `map_func=np.mean`**

转化公式：
```
结构 descriptor = (1/N) × Σ(原子 descriptor_i)  for i = 1 to N
```

具体计算：
```python
# 结构有 20 个原子，每个 35 维
per_atom_desc.shape = (20, 35)

# 平均化
per_structure_desc = np.mean(per_atom_desc, axis=0)
per_structure_desc.shape = (35,)

# 每一维都是对应维度的平均值
per_structure_desc[j] = Σ(atom_i[j]) / 20  for j = 0 to 34
```

---

### 3. PCA 降维到 2D

**位置**: `.venv/lib/python3.13/site-packages/NepTrainKit/core/io/sampler.py` (第 22-66 行)

```python
def pca(X: npt.NDArray[np.float32], n_components: Optional[int] = None):
    """Project a feature matrix onto its leading principal components."""
    # 1. 中心化
    mean = np.mean(X, axis=0)
    X_centered = X - mean
    
    # 2. 计算协方差矩阵
    cov_matrix = np.dot(X_centered.T, X_centered) / (n_samples - 1)
    
    # 3. 特征值分解
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # 4. 按特征值排序
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # 5. 投影到主成分
    X_pca = np.dot(X_centered, eigenvectors[:, :n_components])
    
    return X_pca.astype(np.float32)
```

**应用位置**: `.venv/lib/python3.13/site-packages/NepTrainKit/core/io/base.py` (第 994 行)

```python
# 自动降维到 2D 用于可视化
if reduced.size != 0 and reduced.shape[1] > 2:
    reduced = pca(reduced, 2)
self._descriptor_dataset = NepPlotData(reduced, title="descriptor")
```

**效果**:
- 输入: `(100, 35)` - 100 个结构，35 维 descriptor
- 输出: `(100, 2)` - 100 个结构，2 维坐标
- 保留最大方差方向（通常 > 95%）

---

## 实际数据验证

### 测试数据: RSS.xyz

```
结构数量: 100
每个结构原子数: 20
总原子数: 5000
Descriptor 维度: 35
```

### 数据流转

```
1. 原始数据
   - RSS.xyz: 100 个结构 × 20 个原子 = 2000 个原子

2. NEP 计算 (理论上)
   - Per-atom descriptor: (2000, 35)

3. 聚合操作
   - 对每个结构的 20 个原子取平均
   - Per-structure descriptor: (100, 35)

4. 实际文件
   - descriptor_RSS.out: (100, 35) ✅ 已经是 per-structure

5. PCA 降维
   - 输入: (100, 35)
   - 输出: (100, 2)
   - PC1 解释方差: 96.26%
   - PC2 解释方差: 2.15%
   - 总计: 98.40%

6. 可视化
   - 100 个点在 2D 平面上
   - 每个点代表一个结构
```

---

## 关键问题解答

### Q1: 不同原子数的结构如何统一维度？

**A**: 通过**平均化**操作

```python
# 结构 A: 10 个原子 × 35 维 → 平均 → 35 维
# 结构 B: 50 个原子 × 35 维 → 平均 → 35 维
# 结构 C: 100 个原子 × 35 维 → 平均 → 35 维

# 所有结构都统一到 35 维空间
```

### Q2: 特征矩阵的维度会因原子数不同而不同吗？

**A**: 不会

- **Per-atom descriptor**: 维度固定（如 35 维），与原子数无关
- **Per-structure descriptor**: 通过平均化，维度也固定（35 维）
- **不同的只是原子数量**，不是特征维度

### Q3: 为什么使用简单平均？

**优点**:
- ✅ 简单直观，易于理解和实现
- ✅ 尺寸不变性：不同原子数 → 相同维度
- ✅ 物理意义：代表结构的"平均局部环境"
- ✅ 计算高效：O(N) 复杂度

**局限性**:
- ❌ 丢失原子间相对位置信息
- ❌ 丢失 descriptor 分布信息（方差、极值等）
- ❌ 大小结构的差异被平均化
- ❌ 不同类型原子的贡献被等权重处理

### Q4: 有其他聚合方法吗？

**理论上可以使用**（但 NepTrainKit 未实现）:

```python
# 1. 加权平均（按原子类型或重要性）
weighted_mean = Σ(w_i × descriptor_i) / Σ(w_i)

# 2. 范数（aggregate_per_atom_to_structure 的默认值）
norm = np.linalg.norm(descriptors, axis=0)

# 3. 最大值池化
max_pool = np.max(descriptors, axis=0)

# 4. 求和（保留尺寸信息）
sum_desc = np.sum(descriptors, axis=0)

# 5. 统计特征组合
combined = np.concatenate([
    np.mean(descriptors, axis=0),   # 平均
    np.std(descriptors, axis=0),    # 标准差
    np.max(descriptors, axis=0),    # 最大值
    np.min(descriptors, axis=0)     # 最小值
])  # 维度变为 35 × 4 = 140
```

---

## 代码示例

### 完整流程演示

```python
import numpy as np
from functools import partial

# 模拟数据
atoms_num_list = [20, 15, 18]  # 3 个结构
total_atoms = sum(atoms_num_list)  # 53 个原子
descriptor_dim = 35

# 1. Per-atom descriptor (NEP 输出)
per_atom_desc = np.random.randn(total_atoms, descriptor_dim).astype(np.float32)
print(f"Per-atom descriptor: {per_atom_desc.shape}")  # (53, 35)

# 2. 聚合函数
def split_by_natoms(array, natoms_list):
    counts = np.asarray(list(natoms_list), dtype=int)
    split_indices = np.cumsum(counts)[:-1]
    return np.split(array, split_indices)

def aggregate_per_atom_to_structure(array, atoms_num_list, map_func=np.mean, axis=0):
    split_arrays = split_by_natoms(array, atoms_num_list)
    func = partial(map_func, axis=axis)
    return np.array(list(map(func, split_arrays)))

# 3. 聚合为 per-structure
per_structure_desc = aggregate_per_atom_to_structure(
    per_atom_desc, 
    atoms_num_list, 
    map_func=np.mean, 
    axis=0
)
print(f"Per-structure descriptor: {per_structure_desc.shape}")  # (3, 35)

# 4. PCA 降维
def pca(X, n_components=2):
    mean = np.mean(X, axis=0)
    X_centered = X - mean
    n_samples = X.shape[0]
    cov_matrix = np.dot(X_centered.T, X_centered) / (n_samples - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    X_pca = np.dot(X_centered, eigenvectors[:, :n_components])
    return X_pca.astype(np.float32), eigenvalues

pca_desc, eigenvalues = pca(per_structure_desc, n_components=2)
print(f"PCA descriptor: {pca_desc.shape}")  # (3, 2)

# 5. 可视化坐标
for i, (x, y) in enumerate(pca_desc):
    print(f"结构 {i+1} (原子数={atoms_num_list[i]:2d}): PC1={x:7.3f}, PC2={y:7.3f}")
```

---

## 文件位置索引

### 核心函数

| 函数 | 文件路径 | 行号 |
|------|---------|------|
| `aggregate_per_atom_to_structure` | `NepTrainKit/core/utils.py` | 65-75 |
| `split_by_natoms` | `NepTrainKit/core/utils.py` | 58-65 |
| `pca` | `NepTrainKit/core/io/sampler.py` | 22-66 |
| `get_structures_descriptor` | `NepTrainKit/core/calculator.py` | 311-328 |
| `_load_descriptors` | `NepTrainKit/core/io/base.py` | 963-999 |

### 调用位置

| 调用 | 文件路径 | 行号 | 用途 |
|------|---------|------|------|
| `aggregate_per_atom_to_structure` | `calculator.py` | 326 | 计算 descriptor |
| `aggregate_per_atom_to_structure` | `base.py` | 976 | 加载 descriptor 文件 |
| `aggregate_per_atom_to_structure` | `sampler.py` | 258 | FPS 采样 |
| `pca` | `base.py` | 994 | 降维可视化 |

---

## 总结

1. **NepTrainKit 使用简单平均化处理 per-atom → per-structure 转换**
   - 所有 descriptor 相关调用都使用 `map_func=np.mean`
   - 公式: `结构 descriptor = mean(所有原子 descriptor)`

2. **不同原子数的结构通过平均化统一到相同维度**
   - 每个原子: 35 维 descriptor
   - 每个结构: 35 维 descriptor（原子平均）
   - 维度不变，只是原子数量不同

3. **PCA 降维用于可视化**
   - 从 35D 降到 2D
   - 保留主要方差（通常 > 95%）
   - 每个结构在 2D 平面上是一个点

4. **物理意义**
   - PCA 图上的每个点代表一个结构
   - 点的位置反映结构的"平均局部环境"
   - 相近的点表示结构在 descriptor 空间中相似

---

## 参考资料

- NepTrainKit 源码: `.venv/lib/python3.13/site-packages/NepTrainKit/`
- 验证脚本: `verify_actual_data_corrected.py`
- 演示脚本: `descriptor_pca_flow_demo.py`
