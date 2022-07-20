# Always prefer setuptools over distutils
from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name="binclass-tools",  # 
    
    version="0.2.1",  # Required
    description="A set of tools that facilitates the analysis of binary classification problems",  # Optional
    
    long_description=long_description,  # Optional
    long_description_content_type="text/markdown",  # Optional
   
    url="https://github.com/lucazav/binclass-tools/",  # Optional

    author="Luca Zavarella, Greta Villa",  # Optional

    author_email="lucazavarella@outlook.com",  # Optional

    # For a list of valid classifiers, see https://pypi.org/classifiers/
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Other Audience",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    
    keywords="binary, classification, confusion, matrix, threshold, plot, precision, recall", 
    
    packages=["bctools"], 
    
    python_requires=">=3.6",

    install_requires=["numpy",
                      "pandas",
                      "scikit-learn>=0.22.1",
                      "matplotlib",
                      "plotly"
                     ],  
    
  
    project_urls={  
        "Source": "https://github.com/lucazav/binclass-tools/",
    },
)