"""Application entry point."""

from __future__ import annotations

import sys

from PySide6.QtGui import QSurfaceFormat
from PySide6.QtWidgets import QApplication

from magic_cube.ui.main_window import MainWindow


def main() -> int:
    # The renderer intentionally uses a small fixed-function OpenGL surface;
    # request a compatibility profile before QApplication creates a context.
    surface_format = QSurfaceFormat()
    surface_format.setDepthBufferSize(24)
    surface_format.setVersion(2, 1)
    surface_format.setProfile(QSurfaceFormat.OpenGLContextProfile.CompatibilityProfile)
    QSurfaceFormat.setDefaultFormat(surface_format)

    app = QApplication(sys.argv)
    app.setApplicationName("Magic Cube AI")
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
