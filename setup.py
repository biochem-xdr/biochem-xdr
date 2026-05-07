from setuptools import setup, find_packages

setup(
    name="biochem-xdr",
    version="1.0.0",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "openai>=1.0.0",
        "anthropic>=0.20.0",
        "google-genai>=0.8.0",
        "transformers>=4.42.0",
        "torch>=2.0.0",
        "datasets>=2.18.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "statsmodels>=0.14.0",
        "networkx>=3.1",
        "requests>=2.31.0",
        "matplotlib>=3.8.0",
        "openpyxl>=3.1.0",
        "tqdm>=4.66.0",
        "python-dotenv>=1.0.0",
        "bioservices>=1.11.0",
    ],
)