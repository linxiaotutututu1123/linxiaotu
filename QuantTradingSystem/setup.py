"""
量化交易系统安装脚本
"""
from setuptools import setup, find_packages

# 从 pyproject.toml 读取配置，这里提供向后兼容
setup(
    name="quant-trading-system",
    version="1.0.0",
    packages=find_packages(),
    python_requires=">=3.9",
)
