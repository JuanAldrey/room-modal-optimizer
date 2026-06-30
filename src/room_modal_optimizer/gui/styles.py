from PySide6.QtGui import QPalette, QColor, QFont
from PySide6.QtWidgets import QApplication


STYLESHEET = """
    QPushButton {
        border-radius: 5px;
        padding: 6px 12px;
        font-weight: bold;
        font-size: 10pt;
        border: none;
    }

    QPushButton[role="success"] {
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
            stop:0 #3a9e5f, stop:1 #2d7a49);
        color: #ffffff;
    }
    QPushButton[role="success"]:hover {
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
            stop:0 #4ab870, stop:1 #378f57);
    }
    QPushButton[role="success"]:pressed {
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
            stop:0 #2d7a49, stop:1 #3a9e5f);
    }

    QPushButton[role="secondary"] {
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
            stop:0 #5a5a5a, stop:1 #3d3d3d);
        color: #dddddd;
    }
    QPushButton[role="secondary"]:hover {
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
            stop:0 #6a6a6a, stop:1 #4d4d4d);
    }
    QPushButton[role="secondary"]:pressed {
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
            stop:0 #3d3d3d, stop:1 #5a5a5a);
    }

    QPushButton[role="danger"] {
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
            stop:0 #c0392b, stop:1 #922b21);
        color: #ffffff;
    }
    QPushButton[role="danger"]:hover {
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
            stop:0 #d44637, stop:1 #a93226);
    }
    QPushButton[role="danger"]:pressed {
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
            stop:0 #922b21, stop:1 #c0392b);
    }

    QPushButton[role="link"] {
        background: transparent;
        color: #888888;
        font-size: 8pt;
        font-weight: normal;
        text-decoration: underline;
        border: none;
        padding: 0;
    }
    QPushButton[role="link"]:hover { color: #bbbbbb; }
"""


DARK_PALETTE = {
    QPalette.Window:          QColor(53, 53, 53),
    QPalette.WindowText:      QColor(255, 255, 255),
    QPalette.Base:            QColor(35, 35, 35),
    QPalette.AlternateBase:   QColor(53, 53, 53),
    QPalette.Text:            QColor(255, 255, 255),
    QPalette.Button:          QColor(53, 53, 53),
    QPalette.ButtonText:      QColor(255, 255, 255),
    QPalette.BrightText:      QColor(255, 0, 0),
    QPalette.Highlight:       QColor(42, 130, 218),
    QPalette.HighlightedText: QColor(0, 0, 0),
    QPalette.ToolTipBase:     QColor(255, 255, 220),
    QPalette.ToolTipText:     QColor(0, 0, 0),
    QPalette.Link:            QColor(42, 130, 218),
}

DARK_PALETTE_DISABLED = {
    QPalette.Text:       QColor(128, 128, 128),
    QPalette.ButtonText: QColor(128, 128, 128),
}


def apply_theme(app: QApplication) -> None:
    app.setStyle("Fusion")
    app.setFont(QFont("Segoe UI", 11))

    p = QPalette()
    for role, color in DARK_PALETTE.items():
        p.setColor(role, color)
    for role, color in DARK_PALETTE_DISABLED.items():
        p.setColor(QPalette.Disabled, role, color)
    app.setPalette(p)
    app.setStyleSheet(STYLESHEET)