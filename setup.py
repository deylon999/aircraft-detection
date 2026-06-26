from setuptools import find_packages, setup

setup(
    name="cv-aircraft-detection",
    version="0.1.0",
    description="Сравнение 5 моделей детекции типов самолётов на датасете FGVC-Aircraft",
    packages=find_packages(),
    python_requires=">=3.10",
)
