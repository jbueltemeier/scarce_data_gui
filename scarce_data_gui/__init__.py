try:
    from ._version import version as __version__
except ImportError:
    __version__ = "UNKNOWN"


from . import db, page_parts, streamlit_modules, utils
