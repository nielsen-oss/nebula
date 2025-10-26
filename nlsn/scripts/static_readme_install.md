# nlsn.nebula

This library contains approximately one hundred `PySpark` transformers and a `TransformerPipeline`
class to execute them.<br>
The `TransformerPipeline` is capable of running not only linear pipelines but also branched and nested ones.
Refer to the example notebook for a better understanding.<br>
Additionally, users can add custom transformers to the pipeline as needed.

Although this library is primarily designed for `PySpark` transformations, it also handles
`Pandas` and `Polars` dataframes and their pipelines. However, there are very few transformers
available for `Pandas` and `Polars` (`Count`, `Distinct`, `DropColumns`, `PrintSchema`,
`RenameColumns`, `RoundValues` and `SelectColumns`), so users may need to develop additional
ones if necessary.

Presentation: https://docs.google.com/presentation/d/1TO1pBv37Nf728iSwhqu5zHICvU1JXdgBwKnSaPSpBg8

## Installation
### Using pip
```
python3 -m pip install --upgrade git+https://gitlab+deploy-token-2233151:XroesFyTte_7zyHK4xUK@gitlab.com/nielsen-media/dsci/dsciint/dsci-gps/nebula.git@main#egg=nlsnnebula
```

### In a requirements.txt
Get the latest stable version in your project requirements:
```
git+https://gitlab+deploy-token-2233151:XroesFyTte_7zyHK4xUK@gitlab.com/nielsen-media/dsci/dsciint/dsci-gps/nebula.git@main#egg=nlsnnebula
```

### Development setup

 1. Clone this repo: `git clone git@gitlab.com:nielsen-media/dsci/dsciint/dsci-gps/nebula.git`
 2. With a terminal:
    1. `cd nebula`
    2. `pip install -e .`

### Extra dependencies for specific functionalities

`Nebula` features the graphical representation of the transformer pipelines as a dag.<br>
To render it, `Graphviz` ( [graphviz.org](https://graphviz.org/) ) and `pyyaml`
( [pypi.org/project/PyYAML/](https://pypi.org/project/PyYAML/) ) must be installed on the machine.

( [Graphviz installation](https://graphviz.org/download/) )

The relative `graphviz` package runs under Python 3.8+, to install:
```
pip install graphviz
```
Or:
```
pip install -e ".[graphviz]"
```
