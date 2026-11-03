import numpy as np
import contextlib

if not hasattr(np, '_no_nep50_warning'):
    @contextlib.contextmanager
    def _dummy_no_nep50_warning():
        yield
    np._no_nep50_warning = _dummy_no_nep50_warning

from zipdepth.model.architecture import create_model, ZipDepth
