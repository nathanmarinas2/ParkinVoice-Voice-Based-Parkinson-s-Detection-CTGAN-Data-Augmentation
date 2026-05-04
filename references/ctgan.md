Skip to content
sdv-dev
CTGAN
Repository navigation
Code
Issues
30
 (30)
Pull requests
11
 (11)
Agents
Actions
Projects
Security and quality
Insights
Owner avatar
CTGAN
Public
sdv-dev/CTGAN
Go to file
t
T
Name		
sdv-teamgithub-actions[bot]
sdv-team
and
github-actions[bot]
Automated Latest Dependency Updates (#502)
af2f66e
 · 
3 weeks ago
.github
Add support for Python 3.14 (#483)
4 months ago
ctgan
Bump version: 0.12.1 → 0.12.2.dev0
3 months ago
examples
Parse the right metadata format
7 years ago
scripts
Add support for Python 3.14 (#483)
4 months ago
tests
In verbose mode, make the prefix of the progress bar a fixed-length (#…
3 months ago
.editorconfig
Upgrade cookiecutter and fix lint
7 years ago
.gitignore
Add workflow to release CTGAN on PyPI (#452)
10 months ago
AUTHORS.rst
CTGAN Package Maintenance Updates (#258)
4 years ago
CONTRIBUTING.rst
Update meta info with forum link (#492)
3 months ago
HISTORY.md
v0.12.1 Release Preparation (#496)
3 months ago
LICENSE
CTGAN Package Maintenance Updates (#258)
4 years ago
Makefile
Update release documentation (#453)
10 months ago
README.md
Automated Latest Dependency Updates (#500)
2 months ago
RELEASE.md
Update release guides to include conda-forge step (#498)
3 months ago
latest_requirements.txt
Automated Latest Dependency Updates (#502)
3 weeks ago
pyproject.toml
Bump version: 0.12.1 → 0.12.2.dev0
3 months ago
static_code_analysis.txt
v0.12.1 Release Preparation (#496)
3 months ago
tasks.py
Add support for Python 3.14 (#483)
4 months ago
Repository files navigation
README
Contributing
License

This repository is part of The Synthetic Data Vault Project, a project from DataCebo.

Development Status PyPI Shield Unit Tests Downloads Coverage Status Forum




Overview
CTGAN is a collection of Deep Learning based synthetic data generators for single table data, which are able to learn from real data and generate synthetic data with high fidelity.

Important Links	
💻 Website	Check out the SDV Website for more information about our overall synthetic data ecosystem.
📙 Blog	A deeper look at open source, synthetic data creation and evaluation.
📖 Documentation	Quickstarts, User and Development Guides, and API Reference.
:octocat: Repository	The link to the Github Repository of this library.
⌨️ Development Status	This software is in its Pre-Alpha stage.
👥 DataCebo Forum	Discuss CTGAN features, ask questions, and receive help.
Currently, this library implements the CTGAN and TVAE models described in the Modeling Tabular data using Conditional GAN paper, presented at the 2019 NeurIPS conference.

Install
Use CTGAN through the SDV library
⚠️ If you're just getting started with synthetic data, we recommend installing the SDV library which provides user-friendly APIs for accessing CTGAN. ⚠️

The SDV library provides wrappers for preprocessing your data as well as additional usability features like constraints. See the SDV documentation to get started.

Use the CTGAN standalone library
Alternatively, you can also install and use CTGAN directly, as a standalone library:

Using pip:

pip install ctgan
Using conda:

conda install -c pytorch -c conda-forge ctgan
When using the CTGAN library directly, you may need to manually preprocess your data into the correct format, for example:

Continuous data must be represented as floats
Discrete data must be represented as ints or strings
The data should not contain any missing values
Usage Example
In this example we load the Adult Census Dataset* which is a built-in demo dataset. We use CTGAN to learn from the real data and then generate some synthetic data.

from ctgan import CTGAN
from ctgan import load_demo

real_data = load_demo()

# Names of the columns that are discrete
discrete_columns = [
    'workclass',
    'education',
    'marital-status',
    'occupation',
    'relationship',
    'race',
    'sex',
    'native-country',
    'income',
]

ctgan = CTGAN(epochs=10)
ctgan.fit(real_data, discrete_columns)

# Create synthetic data
synthetic_data = ctgan.sample(1000)
*For more information about the dataset see: Dua, D. and Graff, C. (2019). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and Computer Science.

Join our community
Join our forum to discuss more about CTGAN, ask questions, and receive help.

Interested in contributing to CTGAN? Read our Contribution Guide to get started.

Citing CTGAN
If you use CTGAN, please cite the following work:

Lei Xu, Maria Skoularidou, Alfredo Cuesta-Infante, Kalyan Veeramachaneni. Modeling Tabular data using Conditional GAN. NeurIPS, 2019.

@inproceedings{ctgan,
  title={Modeling Tabular data using Conditional GAN},
  author={Xu, Lei and Skoularidou, Maria and Cuesta-Infante, Alfredo and Veeramachaneni, Kalyan},
  booktitle={Advances in Neural Information Processing Systems},
  year={2019}
}
Related Projects
Please note that these projects are external to the SDV Ecosystem. They are not affiliated with or maintained by DataCebo.

R Interface for CTGAN: A wrapper around CTGAN that brings the functionalities to R users. More details can be found in the corresponding repository: https://github.com/kasaai/ctgan
CTGAN Server CLI: A package to easily deploy CTGAN onto a remote server. Created by Timothy Pillow @oregonpillow at: https://github.com/oregonpillow/ctgan-server-cli



The Synthetic Data Vault Project was first created at MIT's Data to AI Lab in 2016. After 4 years of research and traction with enterprise, we created DataCebo in 2020 with the goal of growing the project. Today, DataCebo is the proud developer of SDV, the largest ecosystem for synthetic data generation & evaluation. It is home to multiple libraries that support synthetic data, including:

🔄 Data discovery & transformation. Reverse the transforms to reproduce realistic data.
🧠 Multiple machine learning models -- ranging from Copulas to Deep Learning -- to create tabular, multi table and time series data.
📊 Measuring quality and privacy of synthetic data, and comparing different synthetic data generation models.
Get started using the SDV package -- a fully integrated solution and your one-stop shop for synthetic data. Or, use the standalone libraries for specific needs.

About
Conditional GAN for generating synthetic tabular data.

Topics
tabular-data generative-adversarial-network data-generation synthetic-data synthetic-data-generation
Resources
 Readme
License
 View license
Contributing
 Contributing
 Activity
 Custom properties
Stars
 1.5k stars
Watchers
 21 watching
Forks
 330 forks
Report repository
Releases 30
v0.12.1
Latest
on Feb 13
+ 29 releases
Deployments
18
 github-pages 6 years ago
+ 17 deployments
Packages
No packages published
Contributors
26
@amontanez24
@csala
@fealho
@sdv-team
@github-actions[bot]
@pvk-developer
@leix28
@katxiao
@R-Palazzo
@rwedge
@frances-h
@kevinykuo
@gsheni
@npatki
+ 12 contributors
Languages
Python
95.2%
 
Makefile
4.8%
Footer
© 2026 GitHub, Inc.
Footer navigation
Terms
Privacy
Security
Status
Community
Docs
Contact
Manage cookies
Do not share my personal information
