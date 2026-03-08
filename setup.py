"""
Setup script for Multi-Modal Robot AI Project
Hệ thống nhận diện đối tượng động 3D sử dụng SNN và camera RGB-D
"""

from pathlib import Path
from setuptools import setup, find_packages

# Đọc README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# Đọc requirements
requirements_file = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_file.exists():
    with requirements_file.open("r", encoding="utf-8") as f:
        requirements = [
            line.strip()
            for line in f
            if line.strip() and not line.startswith("#") and not line.startswith("-")
        ]

setup(
    name="multi-modal-robot-ai",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Hệ thống nhận diện đối tượng động 3D sử dụng SNN và camera RGB-D cho robot thông minh",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/multi_modal_robot_ai",
    packages=find_packages(exclude=["tests", "tests.*", "notebooks", "notebooks.*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
            "isort>=5.12.0",
            "pre-commit>=3.0.0",
        ],
        "jetson": [
            # TensorRT và các dependencies cho Jetson sẽ được cài riêng
            # Xem requirements-jetson.txt
        ],
    },
    entry_points={
        "console_scripts": [
            "robot-ai=main:main",
            "hilo-inspect=scripts.data_collection.hilo_inspect:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
