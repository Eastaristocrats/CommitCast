"""Expand all bundled official source trees without accessing the network."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from fcr.vendor import get_vendor

if __name__ == '__main__':
    for method in ('tafas','petsa','cosa_current'):
        print(get_vendor(method))
