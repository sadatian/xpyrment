import sys
import os
import re
import tomllib
import subprocess
import importlib
import inspect
import pkgutil

def sync_versions(force_pypi_version=None):
    """
    Using pyproject.toml as the absolute single source of truth,
    automatically synchronize the version string across:
    1. src/xpyrment/_version.py
    2. README.md (badges for pypi, release, tests, and coverage)
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
        print(f"📝 Synchronizing {os.path.relpath(version_file_path, root_dir)} -> v{version}")
        with open(version_file_path, "w", encoding="utf-8") as f:
            f.write(expected_version_content)
            
    # 3. Synchronize README.md badges (pypi, release, tests, coverage)
    readme_path = os.path.join(root_dir, "README.md")
    if os.path.exists(readme_path):
        with open(readme_path, "r", encoding="utf-8") as f:
            readme_content = f.read()
            
        # Replace release badge (supporting any digit format)
        updated_content = re.sub(
            r"release-v\d+(?:\.\d+)+(?:%20stable)?",
            f"release-v{version}%20stable",
            readme_content
        )
        # Fetch the latest published version on PyPI or TestPyPI
        pypi_version = None
        try:
            import urllib.request
            import json
            # Try PyPI first
            try:
                with urllib.request.urlopen("https://pypi.org/pypi/xpyrment/json", timeout=3) as r:
                    pypi_version = json.loads(r.read().decode("utf-8"))["info"]["version"]
            except Exception:
                # Fallback to TestPyPI
                with urllib.request.urlopen("https://test.pypi.org/pypi/xpyrment/json", timeout=3) as r:
                    pypi_version = json.loads(r.read().decode("utf-8"))["info"]["version"]
        except Exception:
            pass

        # Replace PyPI badge (supporting any digit format)
        target_pypi_version = force_pypi_version if force_pypi_version else (pypi_version if pypi_version else version)
        print(f"🏷️ Setting PyPI badge to: v{target_pypi_version}")
        updated_content = re.sub(
            r"pypi-v\d+(?:\.\d+)+",
            f"pypi-v{target_pypi_version}",
            updated_content
        )

        
        # 4. Dynamically run pytest and coverage to sync test count and coverage badges
        print("🧪 Running test suite and coverage analysis...")
        try:
            # We use Poetry to execute pytest to ensure lockfile compliance
            cmd = ["poetry", "run", "pytest", "--cov=src", "--cov-report=term"]
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=root_dir, encoding="utf-8")
            
            if result.returncode == 0:
                stdout = result.stdout
                
                # Extract tests passed
                tests_match = re.search(r"(\d+)\s+passed", stdout)
                
                # Extract coverage percentage from TOTAL row
                cov_match = re.search(r"TOTAL\s+\d+\s+\d+\s+(\d+)%", stdout)
                
                if tests_match:
                    passed_count = tests_match.group(1)
                    print(f"✅ Tests: {passed_count} passed.")
                    # Replace tests badge (e.g., tests-138%20passed)
                    updated_content = re.sub(
                        r"tests-\d+%20passed",
                        f"tests-{passed_count}%20passed",
                        updated_content
                    )
                    
                if cov_match:
                    cov_percent = cov_match.group(1)
                    print(f"📊 Coverage: {cov_percent}%")
                    # Replace coverage badge (e.g., coverage-100%25 or coverage-93%25)
                    updated_content = re.sub(
                        r"coverage-\d+%25",
                        f"coverage-{cov_percent}%25",
                        updated_content
                    )
            else:
                print(f"⚠️ Pytest failed with return code {result.returncode}. Skipping badge update.")
                if result.stderr:
                    print(f"Debug Info:\n{result.stderr}")
        except Exception as e:
            print(f"⚠️ Could not run pytest or coverage: {str(e)}")
            pass
        
        if updated_content != readme_content:
            print(f"📝 Synchronizing {os.path.relpath(readme_path, root_dir)} badges...")
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
                        # Clean LaTeX from first line for cleaner rendering in table and escape vertical bars
                        first_line = doc.split("\n")[0] if doc else "No description available."
                        first_line = first_line.replace("$", "").replace("|", "\\|")
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
                first_line = first_line.replace("|", "\\|")
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


if __name__ == "__main__":
    import argparse
    import shutil
    
    

    parser = argparse.ArgumentParser(description="Automate building and publishing the xpyrment package.")
    parser.add_argument("--build", action="store_true", help="Build source distribution and wheel.")
    parser.add_argument("--testpypi", action="store_true", help="Publish the package to TestPyPI.")
    parser.add_argument("--pypi", action="store_true", help="Publish the package to actual PyPI.")
    parser.add_argument("--sync", action="store_true", help="Sync versions and test coverage badges without building.")
    
    args = parser.parse_args()
    
    root_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load .env file variables into environment
    env_path = os.path.join(root_dir, ".env")
    if os.path.exists(env_path):
        try:
            with open(env_path, "r", encoding="utf-8") as env_f:
                for line in env_f:
                    if line.strip() and not line.startswith("#") and "=" in line:
                        k, v = line.strip().split("=", 1)
                        k_clean = k.strip()
                        v_clean = v.strip().strip('"').strip("'")
                        if k_clean not in os.environ:
                            os.environ[k_clean] = v_clean
        except Exception:
            pass
    
    # Check if any argument was passed, if not show help
    if not any(vars(args).values()):
        parser.print_help()
        sys.exit(0)
        
    # Always load single source of truth version at the start
    pyproject_path = os.path.join(root_dir, "pyproject.toml")
    with open(pyproject_path, "rb") as f:
        config = tomllib.load(f)
    version = config["project"]["version"]

    # Ensure everything is in sync before any build or publish action
    if args.build or args.pypi or args.testpypi or args.sync:
        # Check if Poetry CLI is available on PATH
        if not shutil.which("poetry"):
            print("❌ Error: Poetry CLI is not available on PATH.")
            print("   This project utilizes Poetry for dependency resolution, building, and publishing.")
            print("   Please install Poetry (https://python-poetry.org) or add it to your environment variables.")
            sys.exit(1)
            
        print(f"🔄 Synchronizing versions and badges (Target: v{version})...")
        # For build/publish, we force the PyPI badge to the version being released
        force_v = version if (args.build or args.pypi or args.testpypi) else None
        sync_versions(force_pypi_version=force_v)
        
    if args.build:
        print("🧹 Cleaning previous build and dist directories...")
        for folder in ["build", "dist", "src/xpyrment.egg-info"]:
            folder_path = os.path.join(root_dir, folder)
            if os.path.exists(folder_path):
                print(f"   - Removing {folder}/")
                shutil.rmtree(folder_path)
                
        # Automatically clean any legacy temporary sdist build folders (e.g. xpyrment-1.1.2.5)
        for name in os.listdir(root_dir):
            if name.startswith("xpyrment-") and os.path.isdir(os.path.join(root_dir, name)):
                try:
                    print(f"   - Removing legacy folder {name}/")
                    shutil.rmtree(os.path.join(root_dir, name))
                except Exception:
                    pass

                
        print("📦 Building source distribution and wheel packages via Poetry...")
        build_cmd = ["poetry", "build"]
        print(f"📦 Running Poetry build command: {' '.join(build_cmd)}")
        result = subprocess.run(build_cmd, cwd=root_dir)
        if result.returncode == 0:
            print(f"🎉 Build completed successfully. Artifacts saved inside '{os.path.relpath(os.path.join(root_dir, 'dist'), root_dir)}/' directory.")
            if os.path.exists(os.path.join(root_dir, 'dist')):
                for f in os.listdir(os.path.join(root_dir, 'dist')):
                    print(f"   - {f}")
        else:
            print("❌ Build failed.")
            sys.exit(result.returncode)
            
    if args.testpypi or args.pypi:
        dist_dir = os.path.join(root_dir, "dist")
        if not os.path.exists(dist_dir) or not os.listdir(dist_dir):
            print("❌ No distribution files found in 'dist/' directory. Please run with --build flag first.")
            sys.exit(1)
        readme_path = os.path.join(root_dir, "README.md")
        notes_path = os.path.join(root_dir, "RELEASE_NOTES.md")
        
        # Load or generate release notes content
        if os.path.exists(notes_path):
            with open(notes_path, "r", encoding="utf-8") as f:
                notes_content = f.read()
        else:
            notes_content = f"# Release Notes - v{version}\n\nAutomated package release."
            with open(notes_path, "w", encoding="utf-8") as f:
                f.write(notes_content)
        

        # Helper to create GitHub Release
        def create_github_release(v, body):
            tag_name = f"v{v}"
            print(f"🏷️ Creating local Git tag: {tag_name}...")
            # Delete existing tag if any to avoid collision
            subprocess.run(["git", "tag", "-d", tag_name], capture_output=True)
            subprocess.run(["git", "tag", "-a", tag_name, "-m", f"Release {tag_name}\n\n{body}"], check=False)
            
            print(f"🚀 Pushing Git tag {tag_name} to origin...")
            subprocess.run(["git", "push", "origin", f":refs/tags/{tag_name}"], capture_output=True)  # Delete remote tag if any
            subprocess.run(["git", "push", "origin", tag_name], check=False)
            
            # Check GITHUB_TOKEN or GH_TOKEN (automatically loaded globally from .env or system environment)
            token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
            if not token:
                print("⚠️ GITHUB_TOKEN or GH_TOKEN env variables not found. Tag has been pushed, but skipping API release creation.")
                return

            print(f"🔑 GitHub Token found ({'GH_TOKEN' if os.environ.get('GH_TOKEN') else 'GITHUB_TOKEN'}). Proceeding with API release...")


                
            print("🚀 Creating formal GitHub Release via REST API...")
            import urllib.request
            import json
            
            try:
                remote_result = subprocess.run(["git", "remote", "get-url", "origin"], capture_output=True, text=True, check=True)
                remote_url = remote_result.stdout.strip()
                match = re.search(r"github\.com[:/]([^/]+)/([^.]+)", remote_url)
                if match:
                    owner = match.group(1)
                    repo = match.group(2)
                else:
                    owner = "sadatian"
                    repo = "xpyrment"
            except Exception:
                owner = "sadatian"
                repo = "xpyrment"
            
            print(f"📁 Repository: {owner}/{repo}")
                
            api_url = f"https://api.github.com/repos/{owner}/{repo}/releases"
            headers = {
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
                "Content-Type": "application/json"
            }
            
            data = {
                "tag_name": tag_name,
                "target_commitish": "main",
                "name": tag_name,
                "body": body,
                "draft": False,
                "prerelease": False
            }
            
            try:
                req = urllib.request.Request(
                    api_url, 
                    data=json.dumps(data).encode("utf-8"), 
                    headers=headers, 
                    method="POST"
                )
                with urllib.request.urlopen(req) as response:
                    res_data = json.loads(response.read().decode("utf-8"))
                    print(f"🎉 GitHub Release '{tag_name}' created successfully!")
                    print(f"🔗 View Release: {res_data.get('html_url')}")
                    
                    release_id = res_data.get("id")
                    upload_url_template = res_data.get("upload_url")
                    if release_id and upload_url_template:
                        upload_url = upload_url_template.split("{")[0]
                        print(f"📦 Release ID: {release_id}")
                        print(f"📤 Uploading assets to: {upload_url}")
                        if os.path.exists(dist_dir):
                            for filename in os.listdir(dist_dir):
                                file_path = os.path.join(dist_dir, filename)
                                if os.path.isfile(file_path):
                                    print(f"📤 Uploading build asset to GitHub Release: {filename}...")
                                    try:
                                        with open(file_path, "rb") as asset_f:
                                            asset_data = asset_f.read()
                                        asset_req = urllib.request.Request(
                                            f"{upload_url}?name={filename}",
                                            data=asset_data,
                                            headers={
                                                "Authorization": f"Bearer {token}",
                                                "Content-Type": "application/octet-stream",
                                            },
                                            method="POST"
                                        )
                                        urllib.request.urlopen(asset_req)
                                    except Exception as ae:
                                        print(f"⚠️ Asset upload failed for {filename}: {str(ae)}")
            except Exception as e:
                print(f"❌ Failed to create GitHub Release via REST API: {str(e)}")
        
        if args.testpypi:
            print("🚀 Uploading distribution files to TestPyPI via Poetry...")
            env_vars = os.environ.copy()
            pypi_token = env_vars.get("TESTPYPI_TOKEN") or env_vars.get("PYPI_TOKEN")
            
            # Configure TestPyPI repository in Poetry
            subprocess.run(["poetry", "config", "repositories.testpypi", "https://test.pypi.org/legacy/"], check=True)
            
            if pypi_token:
                token_source = "TESTPYPI_TOKEN" if env_vars.get("TESTPYPI_TOKEN") else "PYPI_TOKEN"
                print(f"🔑 Using API Token for TestPyPI authentication ({token_source}).")
                # Pass token via environment to avoid persisting credentials in Poetry config
                env_vars["POETRY_PYPI_TOKEN_TESTPYPI"] = pypi_token
            else:
                print("⚠️ No API Token found for TestPyPI. Poetry may prompt for credentials.")
            
            result = subprocess.run(["poetry", "publish", "-r", "testpypi"], cwd=root_dir, env=env_vars)
            
            if result.returncode != 0:
                print("❌ Upload to TestPyPI failed! Reverting PyPI badge in README.md to latest available version...")
                sync_versions() # This will fetch the latest from PyPI/TestPyPI
            else:
                print("🎉 Successfully uploaded and synchronized PyPI badge!")
                create_github_release(version, notes_content)
            
        if args.pypi:
            print("🚀 Uploading distribution files to actual PyPI via Poetry...")
            env_vars = os.environ.copy()
            pypi_token = env_vars.get("PYPI_TOKEN")
            
            if pypi_token:
                print("🔑 Using API Token for PyPI authentication (PYPI_TOKEN).")
                # Pass token via environment to avoid persisting credentials in Poetry config
                env_vars["POETRY_PYPI_TOKEN_PYPI"] = pypi_token
            else:
                print("⚠️ No PYPI_TOKEN found. Poetry may prompt for credentials.")
                
            result = subprocess.run(["poetry", "publish"], cwd=root_dir, env=env_vars)
            
            if result.returncode != 0:
                print("❌ Upload to PyPI failed! Reverting PyPI badge in README.md to latest available version...")
                sync_versions() # This will fetch the latest from PyPI/TestPyPI
            else:
                print("🎉 Successfully uploaded and synchronized PyPI badge!")
                create_github_release(version, notes_content)



