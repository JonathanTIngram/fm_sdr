"""RDS / packet info window: a standalone window with its own background
thread (rds_worker.RdsWorker), showing station name, RadioText (artist,
song, etc., as free text -- RDS carries no images, that's a hard protocol
limit), and a live log of decoded groups as they resolve.
"""
from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QLabel, QPlainTextEdit, QVBoxLayout, QWidget

from rds_worker import RdsWorker

REFRESH_MS = 200
LOG_DISPLAY_LINES = 16


class RdsWindow(QWidget):
    """Standalone window showing live-decoded RDS station info. Owns its
    own RdsWorker thread; feed it raw IQ blocks from wherever the SDR is
    actually being read via feed_samples()."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("RDS / Packet Info")

        self.station_label = QLabel("Station: (searching…)")
        station_font = self.station_label.font()
        station_font.setPointSize(16)
        station_font.setBold(True)
        self.station_label.setFont(station_font)

        self.radiotext_label = QLabel("RadioText: —")
        self.radiotext_label.setWordWrap(True)

        self.status_label = QLabel("Sync: searching…  |  blocks ok=0 err=0")
        status_font = self.status_label.font()
        status_font.setPointSize(9)
        self.status_label.setFont(status_font)

        log_title = QLabel("Decoded groups (live)")
        bold_font = log_title.font()
        bold_font.setBold(True)
        log_title.setFont(bold_font)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setFont(QFont("Monospace", 9))
        self.log.setToolTip(
            "Every line is one decoded RDS group as it resolves — group "
            "type, which piece of the station name/RadioText it carries, "
            "and the characters themselves."
        )

        layout = QVBoxLayout()
        layout.addWidget(self.station_label)
        layout.addWidget(self.radiotext_label)
        layout.addWidget(self.status_label)
        layout.addWidget(log_title)
        layout.addWidget(self.log)
        self.setLayout(layout)
        self.resize(520, 420)

        self._log_shown = 0

        self.worker = RdsWorker()
        self.worker.start()

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._redraw)
        self._timer.start(REFRESH_MS)

    def feed_samples(self, samples):
        self.worker.feed_samples(samples)

    def reset(self):
        """Call when the tuned frequency changes -- the old station's
        PS/RadioText/log no longer applies."""
        self.worker.reset()
        self._log_shown = 0
        self.log.clear()

    def _redraw(self):
        groups = self.worker.rds.groups
        sync = self.worker.rds.sync

        name = groups.station_name
        self.station_label.setText(f"Station: {name if name else '(none yet)'}")

        rt = groups.radiotext
        pty = groups.pty_name
        rt_display = rt if rt else '—'
        if pty:
            rt_display += f"   [{pty}]"
        self.radiotext_label.setText(f"RadioText: {rt_display}")

        lock_text = "locked" if sync.locked else "searching…"
        self.status_label.setText(
            f"Sync: {lock_text}  |  blocks ok={sync.blocks_ok} err={sync.blocks_error}"
        )

        new_count = groups.group_count - self._log_shown
        if new_count > 0:
            recent = list(groups.log)[-new_count:] if new_count <= len(groups.log) else list(groups.log)
            for group in recent:
                fields = group['fields']
                if fields:
                    line = (
                        f"[{group['group_type']}{group['version']}] "
                        f"PI={group['pi']:04X} {fields['name']}"
                        f"[{fields['segment']:>2}] = {fields['chars']!r}"
                    )
                else:
                    line = f"[{group['group_type']}{group['version']}] PI={group['pi']:04X} (unhandled group type)"
                self.log.appendPlainText(line)
            self._log_shown = groups.group_count

    def closeEvent(self, event):
        self._timer.stop()
        self.worker.stop()
        self.worker.wait(2000)
        event.accept()
