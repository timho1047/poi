"""
DataLoader 模块：为 RQ-VAE 训练构建 PyTorch Dataset 和 DataLoader

功能：
- POIDataset: 从 poi_features.pt 读取 POI 特征张量，并提供索引访问
- get_dataloader: 工厂函数，返回配置好的 DataLoader（支持打乱、多进程、pin_memory 等）

使用示例：
    from DataLoader import get_dataloader
    train_loader = get_dataloader(dataset_path='../Dataset/NYC', batch_size=128, device='cuda')
    for batch in train_loader:
        # batch 为 shape [batch_size, feature_dim] 的 Tensor
        pass
"""

from pathlib import Path
from typing import Literal, Optional, Tuple, Dict, List
import os
import torch
from torch.utils.data import Dataset, DataLoader


class POIDataset(Dataset):
    """
    POI 特征数据集
    
    从指定目录加载 poi_features.pt（Tensor，shape=[num_pois, total_dim]）
    并提供按索引访问的接口。
    
    Args:
        dataset_root: 数据集目录路径，例如 '../Dataset/NYC'
                      该目录下需包含 poi_features.pt
    """
    
    def __init__(self, dataset_root: str, indices: Optional[torch.Tensor] = None):
        features_path = os.path.join(dataset_root, 'poi_features.pt')
        if not os.path.exists(features_path):
            raise FileNotFoundError(f"POI features file not found: {features_path}")
        
        # 一次性加载到内存（适用于数据量不超过内存容量的场景）
        self.features = torch.load(features_path)
        self.indices = indices  # 一个 LongTensor，指向 features 的子集索引；None 表示使用全部
        print(f"[POIDataset] Loaded features from {features_path}, shape: {self.features.shape}")
        
        # 可选：在此处添加数据预处理，例如归一化、裁剪等
        # self.features = (self.features - self.features.mean(dim=0)) / (self.features.std(dim=0) + 1e-8)
    
    def __len__(self):
        """返回数据集中样本总数"""
        if self.indices is not None:
            return self.indices.shape[0]
        return self.features.shape[0]
    
    def __getitem__(self, idx):
        """
        根据索引返回一个样本
        
        Args:
            idx: 样本索引（int 或 tensor）
        
        Returns:
            x: shape [feature_dim] 的 Tensor，表示一个 POI 的特征向量
        """
        if self.indices is not None:
            real_idx = int(self.indices[idx])
            return self.features[real_idx]
        return self.features[int(idx)]


def get_dataloader(
    dataset_path: Path|str,
    batch_size: int = 128,
    shuffle: bool = True,
    num_workers: int = 4,
    device: str = Literal["cpu", "cuda", "mps"],
    drop_last: bool = False,
    prefetch_factor: int = 2
) -> DataLoader:
    """
    创建并返回配置好的 DataLoader
    
    Args:
        dataset_path: 数据集根目录，例如 '../Dataset/NYC'
        batch_size: 每批样本数量
        shuffle: 是否在每个 epoch 开始时随机打乱数据（默认 True）
        num_workers: 数据加载的并行进程数（0=主进程加载；推荐 2~8）
        device: 目标设备（'cpu' / 'cuda' / 'mps'），用于决定是否启用 pin_memory
        drop_last: 是否丢弃最后一个不完整的批次（默认 False）
        prefetch_factor: 每个 worker 预取的批次数（默认 2，仅当 num_workers>0 时生效）
    
    Returns:
        DataLoader 实例
    
    注意：
        - 如果 device='cuda' 且 num_workers>0，会自动启用 pin_memory 以加速 CPU->GPU 数据传输
        - persistent_workers=True 可避免反复创建子进程，适合多 epoch 训练
        - 对于纯内存 Tensor 数据，num_workers 带来的收益有限，但不会有负面影响
    """
    dataset = POIDataset(dataset_root=str(dataset_path))
    
    # 如果使用 CUDA，开启 pin_memory 可加速 H2D（Host to Device）拷贝
    pin_memory = (device == 'cuda')
    
    # persistent_workers: 当 num_workers>0 时复用子进程，避免每个 epoch 重复创建
    persistent = (num_workers > 0)
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=persistent
    )
    
    print(f"[DataLoader] Created with batch_size={batch_size}, shuffle={shuffle}, "
          f"num_workers={num_workers}, pin_memory={pin_memory}, drop_last={drop_last}")
    
    return loader


def _ensure_splits(
    dataset_path: Path | str,
    ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 43,
) -> Dict[str, torch.Tensor]:
    """
    依据给定比例在首次运行时生成 train/val/test 索引，并保存在 <dataset_path>/splits.pt。
    后续运行直接复用该文件，保证划分一致性。
    """
    dataset_path = Path(dataset_path)
    split_file = dataset_path / "splits.pt"

    if split_file.exists():
        splits = torch.load(split_file)
        # 兼容性检查
        if all(k in splits for k in ("train", "val", "test")):
            return {k: torch.as_tensor(v, dtype=torch.long) for k, v in splits.items()}

    features: torch.Tensor = torch.load(dataset_path / "poi_features.pt")
    n = features.shape[0]

    r_train, r_val, r_test = ratios
    assert abs((r_train + r_val + r_test) - 1.0) < 1e-6, "split ratios must sum to 1.0"

    g = torch.Generator()
    g.manual_seed(seed)
    perm = torch.randperm(n, generator=g)

    n_train = int(n * r_train)
    n_val = int(n * r_val)
    n_test = n - n_train - n_val

    idx_train = perm[:n_train]
    idx_val = perm[n_train:n_train + n_val]
    idx_test = perm[n_train + n_val:]

    splits = {"train": idx_train, "val": idx_val, "test": idx_test}
    # 保存为 CPU 上的 list，便于跨设备加载
    torch.save({k: v.cpu() for k, v in splits.items()}, split_file)
    print(f"[Split] Created splits at {split_file} with sizes: ",
          {k: int(v.numel()) for k, v in splits.items()})
    return splits


def get_dataloaders(
    dataset_path: Path | str,
    batch_size: int = 128,
    num_workers: int = 4,
    device: str = Literal["cpu", "cuda", "mps"],
    drop_last: bool = False,
    prefetch_factor: int = 2,
    ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 43,
) -> Dict[str, DataLoader]:
    """
    返回 train/val/test 三个 DataLoader，按给定比例划分，且划分在磁盘持久化。
    - train: shuffle=True
    - val/test: shuffle=False
    """
    splits = _ensure_splits(dataset_path, ratios, seed)

    pin_memory = (device == 'cuda')
    persistent = (num_workers > 0)

    def make_loader(split: str, shuffle_flag: bool) -> DataLoader:
        ds = POIDataset(str(dataset_path), indices=splits[split])
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last if split == 'train' else False,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            persistent_workers=persistent,
        )

    loaders = {
        'train': make_loader('train', True),
        'val': make_loader('val', False),
        'test': make_loader('test', False),
    }
    print("[DataLoaders] Created with sizes:", {k: len(v) for k, v in loaders.items()})
    return loaders
