"""Dynamically create README.md with the collapsible transformers section."""

from collections import defaultdict
from pathlib import Path
from typing import Type

from nlsn.nebula import spark_transformers
from nlsn.nebula.base import Transformer

_DICT_FOLDERS = {
    "arrays": "ArrayType Columns Manipulations.",
    "aggregations": "GroupBy and Window Operations.",
    "assertions": "Assert certain conditions and raise error if necessary.",
    "cleanup": "Data Cleanup and Fixing Utilities.",
    "collection": "General Purposes Transformers.",
    "columns": "Transformers for Managing Columns without Affecting Row Values.",
    "debug": "Transformers for Debugging Purposes.",
    "filters": "Row Filtering Operations.",
    "logs": "Logging and Monitoring Tools.",
    "mappings": "MapType Columns Manipulations.",
    "ml": "Transformers for Machine Learning Features.",
    "numerical": "Numerical Computations.",
    "partitions": "Spark Partition Handling Utilities.",
    "schema": "Dataframe Schema Operations: Casting, Nullability, etc.",
    "strings": "String Manipulations: Concatenation, Regex, Formatting, etc.",
    "tables": "Spark TemporaryView and Nebula Storage Utilities.",
    "temporal": "Time Manipulation, Conversions and Operations.",
}

_EXTRA_REQUIREMENTS = {
    "cpu-info": {
        "packages": ["py-cpuinfo"],
        "transformers": ["CpuInfo"]
    },
    "holidays": {
        "packages": ["holidays>=0.26"],
        "transformers": ["IsHoliday"]
    },
    "pandas": {
        "packages": ["pandas>=1.2.5"],
        "transformers": ["LogDataSkew"]
    },
    "graphviz": {
        "packages": ["graphviz"],
        "functionalities": ["Pipeline rendering"]
    }
}


def get_extra_deps_section():
    """Create the optional dependencies section."""
    ret = "### Extra dependencies for specific transformers\n"
    ret += "The following transformers need some extra dependencies:\n\n"
    ret += "```"
    for dep, nd in sorted(_EXTRA_REQUIREMENTS.items()):
        li_trf = nd.get("transformers")
        if not li_trf:
            continue
        ret += f"\n{dep}:\n"
        ret += f"    packages: {nd['packages']}\n"
        ret += f"    used in transformers: {li_trf}\n"

    ret += "```\n"
    ret += "To install a specific dependency put its name inside square brackets, e.g.:\n\n"
    ret += '`pip install ".[cpu-info]"`\n\n'
    ret += "Or, to install more options, separate the option names with a comma.\n\n"
    ret += 'Eg `pip install ".[cpu-info, holidays]"`\n\n'
    ret += "To install all the packages above use the parameter `full`\n\n"
    ret += '`pip install ".[full]"`\n\n'
    ret += "To install from git, specifying the options, prepend `nlsnnebula[your options] @`\n\n"

    git_link = "git+https://gitlab+deploy-token-2233151:XroesFyTte_7zyHK4xUK@"
    git_link += (
        "gitlab.com/nielsen-media/dsci/dsciint/dsci-gps/nebula.git@main#egg=nlsnnebula"
    )

    ret += f"`nlsnnebula[your options] @ {git_link}`"

    return ret


def get_transformers() -> dict:
    """Find all the transformers and put them in a dict(name -> transformer)."""
    d_all_attrs = {i: getattr(spark_transformers, i) for i in dir(spark_transformers)}

    d_transformers: dict[str, Type[Transformer]] = {}

    for k, v in d_all_attrs.items():
        try:
            if issubclass(v, Transformer):
                d_transformers[k] = v
        except TypeError:
            pass

    print(f"Found {len(d_transformers)} transformers")
    return d_transformers


def get_transformer_data(d_transformers: dict[str, Type[Transformer]]) -> list:
    """Get transformer data and information.

    Args:
        d_transformers: dict(str, Transformer)
            Output of get_transformers()

    Returns: list(tuple(str, str, str, bool))
        Tuple elements:
        - transformer name (eg: AddTypedColumns)
        - transformer folder (eg: manipulation)
        - transformer docstring
        - is deprecated (bool)
        The outer list is sorted by transformer name
    """
    li_data = []
    trf: Type[Transformer]

    for name, trf in d_transformers.items():
        docs = trf.__init__.__doc__

        # if name == "LagOverWindow":
        #     print(docs)

        if not docs:
            print(f"No docs found for {name}")

        module_name: str = trf.__module__.split(".")[-1]
        deprecated: bool = getattr(trf, "DEPRECATED", False)
        li_data.append((name, module_name, docs, deprecated))

    return sorted(li_data)


def to_collapsible(title: str, text: str) -> str:
    """Create a collapsible markdown element."""
    return f"<details>\n<summary>{title}\n</summary>\n\n{text}\n</details>"


def to_inner_collapsible(o: tuple[str, str, str, bool]) -> dict:
    """Create a collapsible paragraph describing the transformer."""
    name, folder, docstring, deprecated = o
    header = docstring.split(".\n")[0] + "."
    # print(name, folder)
    # print(header)
    text = docstring.strip().replace("`", "'")

    text = [i for i in text.split("\n") if i]
    new_text = ["```" + text[0] + "```<br>", "<br>"]

    for line in text[1:]:
        line = line[8:]
        counter = 0
        for c in line:
            if c != " ":
                break
            counter += 1

        new_line = "```" + line[counter:] + "```<br>"
        new_line = "&nbsp;" * counter * 2 + new_line
        new_text.append(new_line)

    new_text = "".join(new_text)

    if deprecated:
        name_composed = f"{name} *DEPRECATED*"
    else:
        name_composed = name

    title = f"<b>{name_composed} </b> (<i>{folder}</i>)<b>:</b> {header}"
    # create a collapsible cell of the transformer
    cell = to_collapsible(title, new_text)

    ret = {
        "folder": folder,
        "name": name,
        "header": title,
        "text": new_text,
        "cell": cell,
    }

    return ret


def to_outer_collapsible(cells):
    """Create a collapsible paragraph describing the logical division."""
    dict_cell = defaultdict(list)

    for el in cells:
        folder = el["folder"]
        dict_cell[folder].append(el)

    # dict_cell = {k: sorted(v, key=lambda x: x["name"]) for k, v in dict_cell.items()}

    ret = []
    n_tot = 0  # total number of transformers

    for folder, nd in sorted(dict_cell.items()):
        sorted_li = sorted(nd, key=lambda x: x["name"])
        n = len(sorted_li)  # number of transformers for a given folder
        n_tot += n
        # alphabetical sort
        sorted_cells = [i["cell"] for i in sorted_li]
        title = f"<b>{folder} ({n}):</b> ".upper() + _DICT_FOLDERS[folder.lower()]
        collapsible = to_collapsible(title, "<br>".join(sorted_cells))
        ret.append(collapsible)

    return "<br>".join(ret), n_tot


def create_dynamic_transformers_section():
    """Create the dynamic transformer paragraph to append."""
    d_transformers = get_transformers()

    li_transf_data = get_transformer_data(d_transformers)

    cells = [to_inner_collapsible(i) for i in li_transf_data]

    return to_outer_collapsible(cells)


def read_static_install() -> str:
    """Read the static markdown in this folder."""
    with open("static_readme_install.md", "r", encoding="utf-8") as f:
        static_md = f.read()
    return static_md


def read_static_standards() -> str:
    """Read the static markdown in this folder."""
    with open("static_readme_standards.md", "r", encoding="utf-8") as f:
        static_md = f.read()
    return static_md


def main():
    """Write out the full README.md."""
    md: str = read_static_install()
    md += "\n" + get_extra_deps_section() + "\n"
    md += "\n" + read_static_standards() + "\n"

    transformers_section, n_tot = create_dynamic_transformers_section()

    transformer_title: str = f"\n## Transformers ({n_tot})\n"
    md += transformer_title + transformers_section + "\n"

    outer_folder = Path(".").absolute().parent.parent
    path = str(outer_folder / "README.md")

    with open(path, "w", encoding="utf-8") as f:
        f.write(md)


if __name__ == "__main__":
    main()
