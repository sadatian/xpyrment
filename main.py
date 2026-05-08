import sys
import os
import re
import tomllib
import subprocess
import importlib
import inspect
import pkgutil

def sync_versions():
    """
    Using pyproject.toml as the absolute single source of truth,
    automatically synchronize the version string across:
    1. src/xpyrment/_version.py
    2. README.md
    """
    root_dir = os.path.dirname(__file__)
    
    # 1. Read version from pyproject.toml
    pyproject_path = os.path.join(root_dir, "pyproject.toml")
    with open(pyproject_path, "rb") as f:
        config = tomllib.load(f)
    version = config["project"]["version"]
    
    # 2. Synchronize src/xpyrment/_version.py
    version_file_path = os.path.join(root_dir, "src", "xpyrment", "_version.py")
    expected_version_content = f'__version__ = "{version}"\n'
    
    current_version_content = ""
    if os.path.exists(version_file_path):
        with open(version_file_path, "r", encoding="utf-8") as f:
            current_version_content = f.read()
            
    if current_version_content != expected_version_content:
        with open(version_file_path, "w", encoding="utf-8") as f:
            f.write(expected_version_content)
            
    # 3. Synchronize README.md release & PyPI badges
    readme_path = os.path.join(root_dir, "README.md")
    if os.path.exists(readme_path):
        with open(readme_path, "r", encoding="utf-8") as f:
            readme_content = f.read()
            
        # Replace release badge (e.g., release-v1.1.0.1%20stable)
        updated_content = re.sub(
            r"release-v\d+\.\d+\.\d+\.\d+%20stable",
            f"release-v{version}%20stable",
            readme_content
        )
        # Replace PyPI badge (e.g., pypi-v1.1.0.1)
        updated_content = re.sub(
            r"pypi-v\d+\.\d+\.\d+\.\d+",
            f"pypi-v{version}",
            updated_content
        )
        
        if updated_content != readme_content:
            with open(readme_path, "w", encoding="utf-8") as f:
                f.write(updated_content)
                
    return version

def get_doe_designs():
    """
    Discovers all DoE classes dynamically from the xpyrment.design.doe package.
    """
    designs = []
    try:
        root_dir = os.path.dirname(__file__)
        src_path = os.path.join(root_dir, "src")
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
            
        import xpyrment.design.doe as doe_pkg
        pkg_path = os.path.dirname(doe_pkg.__file__)
        
        for _, module_name, _ in pkgutil.iter_modules([pkg_path]):
            if module_name in ("base", "__init__"):
                continue
            try:
                mod = importlib.import_module(f"xpyrment.design.doe.{module_name}")
                for name, obj in inspect.getmembers(mod, inspect.isclass):
                    # Ensure the class belongs to this module and resembles a design model
                    if (obj.__module__ == mod.__name__ and 
                        ("Design" in name or name.endswith("Factorial") or name.endswith("Composite") or name == "EVOP")):
                        doc = inspect.getdoc(obj) or ""
                        first_line = doc.split("\n")[0] if doc else "No description available."
                        # Clean LaTeX from first line for cleaner rendering in table
                        first_line = first_line.replace("$", "")
                        designs.append({
                            "name": name,
                            "module": f"xpyrment.design.doe.{module_name}",
                            "description": first_line
                        })
            except Exception:
                continue
    except Exception:
        pass
    return sorted(designs, key=lambda x: x["name"])

def get_metrics():
    """
    Discovers all Metric classes dynamically from xpyrment.metrics.taxonomy.
    """
    metrics = []
    try:
        root_dir = os.path.dirname(__file__)
        src_path = os.path.join(root_dir, "src")
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
            
        import xpyrment.metrics.taxonomy as tax_mod
        for name, obj in inspect.getmembers(tax_mod, inspect.isclass):
            if obj.__module__ == tax_mod.__name__ and not name.startswith("Base"):
                doc = inspect.getdoc(obj) or ""
                first_line = doc.split("\n")[0] if doc else "No description available."
                metrics.append({
                    "name": name,
                    "description": first_line
                })
    except Exception:
        pass
    return sorted(metrics, key=lambda x: x["name"])

def define_env(env):
    """
    Custom MkDocs-Macros hook to automate all versioning, code inclusion,
    dynamic python reflection, and live console outputs.
    """
    # 1. Version Synchronization
    version = sync_versions()
    env.variables['version'] = version
    env.conf['repo_name'] = f"xpyrment v{version}"
    
    # 2. Reflection: Dynamic DoE Class Directory Macro
    @env.macro
    def list_doe_designs():
        """
        Dynamically generates a premium markdown table of all active DoE designs
        discovered inside the package, complete with description summaries.
        """
        designs = get_doe_designs()
        if not designs:
            return "*No DoE design models found.*"
        
        lines = [
            "| Experimental Design | Technical Reference | Key Characteristics & Methodology |",
            "| :--- | :--- | :--- |"
        ]
        for d in designs:
            lines.append(f"| **{d['name']}** | `xpyrment.design.doe.{d['name']}` | {d['description']} |")
        return "\n".join(lines)
        
    # 3. Reflection: Dynamic Metric Class Directory Macro
    @env.macro
    def list_metrics():
        """
        Dynamically generates a premium markdown table of all metric models
        available for statistical variance reduction and Delta method calculations.
        """
        metrics = get_metrics()
        if not metrics:
            return "*No metric taxonomy classes found.*"
            
        lines = [
            "| Metric Model | Technical Class Path | Key Features & Analytical Properties |",
            "| :--- | :--- | :--- |"
        ]
        for m in metrics:
            lines.append(f"| **{m['name']}** | `xpyrment.metrics.taxonomy.{m['name']}` | {m['description']} |")
        return "\n".join(lines)

    # 4. Verified Code Snippet Import Macro
    @env.macro
    def code_include(filepath, start_line=None, end_line=None, lang="python"):
        """
        Inserts a precise line-range from any workspace file as a syntax-highlighted block.
        Perfect for pulling real, working test cases as documentation examples.
        """
        try:
            root_dir = os.path.dirname(__file__)
            full_path = os.path.join(root_dir, filepath)
            if not os.path.exists(full_path):
                return f"*Error: Code file {filepath} not found at {full_path}*"
                
            with open(full_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
                
            if start_line is not None or end_line is not None:
                start = (start_line - 1) if start_line is not None else 0
                end = end_line if end_line is not None else len(lines)
                selected_lines = lines[start:end]
            else:
                selected_lines = lines
                
            content = "".join(selected_lines).rstrip()
            return f"```{lang}\n{content}\n```"
        except Exception as e:
            return f"*Error including code file '{filepath}': {str(e)}*"

    # 5. Live Subprocess CLI Command Output Macro
    @env.macro
    def cli_help(command_args):
        """
        Spawns the real xpyrment CLI locally, captures its stdout, and renders it.
        Guarantees that documentation usage commands are never out of sync!
        """
        try:
            root_dir = os.path.dirname(__file__)
            env_vars = os.environ.copy()
            # Ensure workspace src is first in path
            env_vars["PYTHONPATH"] = os.path.join(root_dir, "src")
            
            cmd = [sys.executable, "-m", "xpyrment.cli"] + command_args.split()
            result = subprocess.run(cmd, capture_output=True, text=True, env=env_vars, encoding="utf-8")
            
            output = result.stdout or result.stderr
            output_clean = output.strip()
            
            return f"```console\n$ xpyrment {command_args}\n{output_clean}\n```"
        except Exception as e:
            return f"*Error running live CLI 'xpyrment {command_args}': {str(e)}*"

    # 6. Uniform Premium Badge Generator Macro
    @env.macro
    def badge(name, status, color="blue", logo=None):
        """
        Generates a premium, styled-consistent flat shield following labelColor rules.
        """
        logo_param = f"&logo={logo}" if logo else ""
        logo_color_param = "&logoColor=white" if logo else ""
        return f'<img src="https://img.shields.io/badge/{name}-{status}-{color}?style=flat{logo_param}{logo_color_param}&labelColor=0b0b0b" alt="{name}" />'
