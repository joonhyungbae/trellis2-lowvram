"""Print one message: `python -m trellis2_lowvram.i18n KEY [name=value ...]`.

Used by the install scripts, which have no other way to reach the translated text.
"""

import sys

from . import t

key, *pairs = sys.argv[1:]
print(t(key, **dict(p.split("=", 1) for p in pairs if "=" in p)))
