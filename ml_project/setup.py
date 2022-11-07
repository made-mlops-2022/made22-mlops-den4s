from setuptools import find_packages, setup

setup(name="ml_project",
      packages=find_packages(),
      version="0.1.0",
      description="made22_homework1",
      author="Den4S",
      install_requires=["marshmallow-dataclass==8.5.9",
                        "pandas==1.5.0",
                        "pandas-profiling==2.11.0",
                        "pytest==7.1.3",
                        "scikit-learn==1.0.2",
                        "numpy==1.23.3",
                        "pyyaml==6.0",
                        "click==8.0.4",
                        "matplotlib==3.5.2",
                        "seaborn==0.11.2"]
      )
