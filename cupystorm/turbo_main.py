# turbo_main_mini.py
from pathlib import Path
import sys

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

from cupystorm.turbo_gui import MainWindow
from cupystorm.turbo_wrapper import DnsSimulator


def main() -> None:
    app = QApplication(sys.argv)

    icon_path = Path(__file__).with_name("cupystorm.icns")
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))

    sim = DnsSimulator(n=256)
    sim.step(1)

    window = MainWindow(sim)

    screen = app.primaryScreen().availableGeometry()
    g = window.geometry()
    g.moveCenter(screen.center())
    window.setGeometry(g)

    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
