# AI4ScaDa generalised Artificial Intelligence Workflow for Scarce Data

The innovation project _AI4ScaDa_ of the _it's OWL_ cluster of excellence is dedicated to the research and development of special AI techniques for profitable use in scenarios with little data, also known as _Scarce Data_. While the term _Big Data_ describes a large amount of available and diverse data, Scarce Data refers to a limited amount or incomplete data sets, as they are often collected in laboratories or during experimental investigations. This challenge particularly affects SMEs, but larger companies can also be confronted with similar data problems.

A key aspect of scarce data is that many experiments and data sets are conducted under similar conditions, which means that important variance in the data is missing. However, this variance is crucial to be able to analyse dependencies between input and output variables. To address this problem, _AI4ScaDa_ aims to develop methods and tools that enable the successful use of AI methods even in data-poor environments.

## Purpose of the generalised AI Workflow

The targeted collection and analysis of data is essential for companies in order to optimise processes and retain valuable expert knowledge. This repository presents an AI-supported workflow that supports companies in gaining precise insights with little data effort. The possible applications are seen in two central application areas:

1. **capture of process knowledge:** The optimisation of processes through targeted data collection
2. **preservation of expert knowledge:** The systematic collection and consolidation of expert knowledge for long-term use

# Documentation

WIP

## License

This project is licensed under the [MIT License](LICENSE).

This repository has been developed as part of the _AI4ScaDa_ project. The content provided in this repository is for informational purposes only and is intended to be used at your own risk. We make no representations or warranties of any kind regarding the completeness, accuracy, reliability, or availability of the information contained in this repository.

By using this repository, you acknowledge that any reliance on such information is strictly at your own risk. In no event shall the authors or contributors of this repository be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from or related to your use of the repository.

Please treat this repository with caution and use it responsibly. While it is available for public use, we advise users to thoroughly review and understand the code and its implications before implementation.

# Installation

To start working on `scarce_data_gui` clone the repository from GitLab and set up the development environment with virtualenv:

```shell
git clone 
cd scarce_data_gui
virtualenv .venv
source .venv/bin/activate or .venv\Scripts\activate (on Windows)
pip install .
```

Die streamlit app kann mit folgendem Befehl gestartet werden:
```shell
python -m streamlit run streamlit_pages/Home.py
# oder mit python3
python3 -m streamlit run streamlit_pages/Home.py
```

# Decision Tree Visualisation

This project visualises decision trees using **Graphviz**. To ensure proper functionality, **Graphviz must be installed**. You can download it from the following link:

➡ [Graphviz Download](https://graphviz.org/download/)

### Installation Instructions:

1. Download and install Graphviz.
2. Make sure it is located in: "C:/Program Files/Graphviz/bin"
3. If Graphviz is installed elsewhere, update the corresponding path in the code.

### Alternative Visualisation:

If Graphviz is not installed or configured correctly, a **simplified plot using Matplotlib** will be generated. However, this visualisation may not be as accurate or detailed as the Graphviz version.

For best results, installing and using Graphviz is recommended.

# Funding

This work was partly funded by the Ministry of Economic Affairs, Innovation, Digitalisation and Energy of the State of North Rhine-Westphalia (MWIDE) within the project AI4ScaDa, grant number 005-2111-0016.
![Alternativtext](streamlit_pages/images/Demonstratoren_Foerderhinweis.png)