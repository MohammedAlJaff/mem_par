from pathlib import Path

from setuptools import find_packages, setup


def read_requirements() -> list[str]:
    # Read requirements from requirements.txt
    # Exclude editable installs and comments
    requirements_path = Path(__file__).with_name("requirements.txt")
    requirements = []
    for raw_line in requirements_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("-e ") or line.startswith("--"):
            continue
        requirements.append(line)
    return requirements


setup(
    name="memparam",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=read_requirements(),
)
