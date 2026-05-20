from pathlib import Path
from typing import TYPE_CHECKING

# Help static type checkers / linters detect streamlit while keeping a
# runtime-safe import for environments where streamlit isn't installed.
if TYPE_CHECKING:
    import streamlit as st  # pragma: no cover
else:
    try:
        import streamlit as st
    except ImportError:
        # Helpful message for VS Code users: do not raise to allow the file to be opened
        # in environments where streamlit is not installed. Install via: pip install streamlit
        print("streamlit is not installed. Install it with: pip install streamlit")
        st = None  # type: ignore
import pandas as pd
import numpy as np
import sklearn
import joblib
