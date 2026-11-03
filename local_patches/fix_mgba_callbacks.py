"""Build-time fix for an uninitialized-pointer bug in mGBA's Python bindings.

mCorePythonCallbackCreate (mGBA src/platform/python/core.c, compiled into the
PyPI mgba==0.10.2 wheel's _pylib.abi3.so; still unfixed on mGBA master) mallocs
a struct mCoreCallbacks and sets only 5 of its 8 function pointers. shutdown,
savedataUpdated and alarm keep whatever bytes malloc returned. GBAFrameEnded
(src/gba/gba.c) calls savedataUpdated when an in-game save finishes writing
flash, so non-NULL garbage there makes the emulator jump to a random address on
SAVE (SIGSEGV/SIGBUS; how often depends on heap contents). Proven with ASan,
whose 0xbe malloc fill makes the unpatched bindings crash on every save.

The fix nulls the three pointers in CoreCallbacks.__init__, right after the
struct is created and before Core.__init__ copies it into the core. The build
fails if the bindings don't look exactly as expected or the fix doesn't take.
"""
import importlib.util
import sys
from pathlib import Path

ANCHOR = "        self.context = lib.mCorePythonCallbackCreate(self._handle)\n"
FIX = (
    "        # mCorePythonCallbackCreate leaves these uninitialized (malloc garbage).\n"
    "        self.context.shutdown = ffi.NULL\n"
    "        self.context.savedataUpdated = ffi.NULL\n"
    "        self.context.alarm = ffi.NULL\n"
)

core_py = Path(importlib.util.find_spec("mgba").origin).parent / "core.py"
source = core_py.read_text()
if FIX in source:
    print("mgba callbacks fix already applied")
else:
    if source.count(ANCHOR) != 1:
        sys.exit(f"{core_py}: expected exactly one mCorePythonCallbackCreate anchor")
    core_py.write_text(source.replace(ANCHOR, ANCHOR + FIX))
    print(f"patched {core_py}")

# Imported only now, after core.py is patched, so the check runs the patched code.
from mgba._pylib import ffi
from mgba.core import CoreCallbacks

callbacks = CoreCallbacks()
for field in ("shutdown", "savedataUpdated", "alarm"):
    if getattr(callbacks.context, field) != ffi.NULL:
        sys.exit(f"mCoreCallbacks.{field} is not NULL after the fix")
print("mgba CoreCallbacks: shutdown/savedataUpdated/alarm are NULL")
