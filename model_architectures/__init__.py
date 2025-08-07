# model_architectures/__init__.py

# 只导出 ResNet20，其他模块不要在这里导入
from .resnet_cifar import ResNet20

__all__ = ['ResNet20']