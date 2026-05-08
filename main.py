import sys
import os

def define_env(env):
    """
    Custom MkDocs-Macros hook to automatically read and expose the repository's
    latest version directly from the source code.
    """
    # Insert package src path to read the active version
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
    from xpyrment._version import __version__
    
    # Expose 'version' to be rendered as {{ version }} on any MkDocs markdown page
    env.variables['version'] = __version__
    
    # Dynamically update the top bar repo link name to include the current version!
    env.conf['repo_name'] = f"xpyrment v{__version__}"
