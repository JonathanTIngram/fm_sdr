"""RDS / packet info window: a standalone window with its own background
thread (rds_worker.RdsWorker), showing station name, RadioText (artist,
song, etc., as free text -- RDS carries no images, that's a hard protocol
limit), a live packet-anatomy diagram, and a live log of decoded groups as
they resolve.
"""
import time

from PyQt6.QtCore import QRectF, Qt, QTimer
from PyQt6.QtGui import QColor, QFont, QPainter, QPen
from PyQt6.QtWidgets import QLabel, QPlainTextEdit, QVBoxLayout, QWidget

from rds_worker import RdsWorker

REFRESH_MS = 200
LOG_DISPLAY_LINES = 16
FLASH_SECONDS = 1.2  # how long a diagram field stays highlighted after an update

BG = QColor(15, 15, 20)          # matches fm_spectrum.py's SpectrumBarWidget
NEUTRAL = QColor(215, 220, 215)
MUTED = QColor(125, 135, 130)
DIM = QColor(60, 66, 62)
TEAL = QColor(99, 179, 166)
TEAL_FILL = QColor(99, 179, 166, 40)
AMBER = QColor(240, 169, 74)
AMBER_FILL = QColor(240, 169, 74, 45)
AMBER_GLOW = QColor(240, 169, 74, 130)

DIAGRAM_SIZE = (900, 470)


class PacketAnatomyWidget(QWidget):
    """Live diagram of one RDS group's block/field structure -- the same
    A/B/C/D block layout and Block-B field breakdown you'd draw on a
    whiteboard, except every field lights up with its actual decoded value
    the moment a group carrying it resolves, and fades over FLASH_SECONDS.
    Whichever of the two group-2/group-0 field layouts was most recently
    seen is drawn active; the other stays dim, showing which branch of the
    protocol is actually in use right now."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(*DIAGRAM_SIZE)
        self._flash = {}
        self._values = {}

    def note_group(self, entry):
        now = time.monotonic()
        fields = entry['fields']

        self._values['pi'] = entry['pi']
        self._values['group_type'] = entry['group_type']
        self._values['version'] = entry['version']
        self._values['tp'] = entry.get('tp')
        self._values['pty'] = entry.get('pty')
        for key in ('A', 'B', 'C', 'D', 'group_type', 'ver', 'tp', 'pty'):
            self._flash[key] = now

        if fields.get('name') == 'PS':
            self._values['fork'] = 'ps'
            self._values['d_chars'] = fields['chars']
            self._values['ta'] = fields['ta']
            self._values['ms'] = fields['ms']
            self._values['di'] = fields['di']
            self._values['ps_addr'] = fields['segment']
            for key in ('group_specific', 'ta', 'ms', 'di', 'ps_addr'):
                self._flash[key] = now
        elif fields.get('name') == 'RT':
            self._values['fork'] = 'rt'
            self._values['d_chars'] = fields['chars']
            old_text_ab = self._values.get('text_ab')
            self._values['text_ab'] = fields['text_ab']
            self._values['rt_addr'] = fields['segment']
            for key in ('group_specific', 'text_ab', 'rt_addr'):
                self._flash[key] = now
            if old_text_ab is not None and fields['text_ab'] != old_text_ab:
                self._flash['text_ab_toggle'] = now

        self.update()

    def _alpha(self, key):
        t = self._flash.get(key)
        if t is None:
            return 0.0
        age = time.monotonic() - t
        return max(0.0, 1.0 - age / FLASH_SECONDS)

    def _cell(self, painter, x, y, w, h, top, bottom, border, fill=None, flash_key=None, bottom_color=None):
        rect = QRectF(x, y, w, h)
        painter.setPen(QPen(border, 1.3))
        painter.setBrush(fill if fill is not None else Qt.BrushStyle.NoBrush)
        painter.drawRect(rect)

        if flash_key is not None:
            frac = self._alpha(flash_key)
            if frac > 0.02:
                glow = QColor(AMBER_GLOW)
                glow.setAlpha(int(AMBER_GLOW.alpha() * frac))
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(glow)
                painter.drawRect(rect)
                painter.setPen(QPen(AMBER, 1.8))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawRect(rect)

        top_rect = QRectF(x + 2, y + 3, w - 4, h * 0.55)
        font = QFont("Monospace", 9)
        painter.setFont(font)
        painter.setPen(NEUTRAL)
        painter.drawText(top_rect, int(Qt.AlignmentFlag.AlignCenter), top)

        if bottom:
            bottom_rect = QRectF(x + 2, y + h * 0.55, w - 4, h * 0.42)
            painter.setFont(QFont("Monospace", 8))
            painter.setPen(bottom_color or MUTED)
            painter.drawText(bottom_rect, int(Qt.AlignmentFlag.AlignCenter), bottom)

    def _dashed(self, painter, x1, y1, x2, y2, color=AMBER):
        pen = QPen(color, 1.1)
        pen.setStyle(Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.drawLine(int(x1), int(y1), int(x2), int(y2))

    def _label(self, painter, x, y, w, text, color, size=11, bold=False):
        font = QFont("Monospace", size)
        font.setBold(bold)
        painter.setFont(font)
        painter.setPen(color)
        painter.drawText(QRectF(x, y, w, 20), int(Qt.AlignmentFlag.AlignCenter), text)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), BG)

        v = self._values
        fork = v.get('fork')

        # --- Row 1: four top-level blocks ---
        pi = v.get('pi')
        self._label(painter, 20, 20, 204, "A — PI Code", NEUTRAL, bold=True)
        self._cell(painter, 20, 60, 126, 70, "PI Code", f"{pi:04X}" if pi is not None else "16 bits",
                   NEUTRAL, flash_key='A', bottom_color=AMBER if pi is not None else MUTED)
        self._cell(painter, 146, 60, 78, 70, "check", "10 bits", TEAL, fill=TEAL_FILL)

        self._label(painter, 238, 20, 204, "B — Type & Flags", AMBER, bold=True)
        self._cell(painter, 238, 60, 126, 70, "expanded ↓", "16 bits", AMBER, fill=AMBER_FILL, flash_key='B')
        self._cell(painter, 364, 60, 78, 70, "check", "10 bits", TEAL, fill=TEAL_FILL)

        c_desc = "AF codes /\nPI repeat / text"
        self._label(painter, 456, 20, 204, "C — Varies", NEUTRAL, bold=True)
        self._cell(painter, 456, 60, 126, 70, c_desc, "16 bits", NEUTRAL, flash_key='C')
        self._cell(painter, 582, 60, 78, 70, "check", "10 bits", TEAL, fill=TEAL_FILL)

        d_val = v.get('d_chars')
        self._label(painter, 674, 20, 204, "D — Data", NEUTRAL, bold=True)
        self._cell(painter, 674, 60, 126, 70, "2 characters", repr(d_val) if d_val else "16 bits",
                   NEUTRAL, flash_key='D', bottom_color=AMBER if d_val else MUTED)
        self._cell(painter, 800, 60, 78, 70, "check", "10 bits", TEAL, fill=TEAL_FILL)

        # --- zoom lines from Block B down to its field breakdown ---
        self._dashed(painter, 238, 130, 20, 210)
        self._dashed(painter, 364, 130, 880, 210)

        self._label(painter, 20, 188, 860, "Block B's 16 data bits, expanded", AMBER, bold=True)

        group_type = v.get('group_type')
        version = v.get('version')
        gt_bottom = f"{group_type}{version}" if group_type is not None else "4 bits"
        self._cell(painter, 20, 210, 215, 70, "Group Type", gt_bottom, NEUTRAL,
                   flash_key='group_type', bottom_color=AMBER if group_type is not None else MUTED)

        ver_bottom = version if version else "1 bit"
        self._cell(painter, 235, 210, 53.75, 70, "Ver.", ver_bottom, NEUTRAL,
                   flash_key='ver', bottom_color=AMBER if version else MUTED)

        tp = v.get('tp')
        tp_bottom = str(tp) if tp is not None else "1 bit"
        self._cell(painter, 288.75, 210, 53.75, 70, "TP", tp_bottom, NEUTRAL,
                   flash_key='tp', bottom_color=AMBER if tp is not None else MUTED)

        pty = v.get('pty')
        pty_bottom = str(pty) if pty is not None else "5 bits"
        self._cell(painter, 342.5, 210, 268.75, 70, "Program Type (PTY)", pty_bottom, NEUTRAL,
                   flash_key='pty', bottom_color=AMBER if pty is not None else MUTED)

        self._cell(painter, 611.25, 210, 268.75, 70, "Group-specific", "5 bits — forks below",
                   AMBER, fill=AMBER_FILL, flash_key='group_specific')

        # --- fork: the group-specific 5 bits mean different things per group type ---
        self._dashed(painter, 611.25, 280, 460, 335)
        self._dashed(painter, 880, 280, 880, 335)

        ps_active = fork == 'ps'
        rt_active = fork == 'rt'
        ps_border = NEUTRAL if ps_active else DIM
        rt_border = NEUTRAL if rt_active else DIM
        ps_label_color = NEUTRAL if ps_active else DIM
        rt_label_color = NEUTRAL if rt_active else DIM

        self._label(painter, 445, 300, 230, "Group type 0 · station name", ps_label_color, size=9, bold=True)
        ta = v.get('ta') if ps_active else None
        ms = v.get('ms') if ps_active else None
        di = v.get('di') if ps_active else None
        ps_addr = v.get('ps_addr') if ps_active else None
        self._cell(painter, 460, 340, 40, 65, "TA", str(ta) if ta is not None else "1b", ps_border,
                   flash_key='ta' if ps_active else None, bottom_color=AMBER if ta is not None else MUTED)
        self._cell(painter, 500, 340, 40, 65, "MS", str(ms) if ms is not None else "1b", ps_border,
                   flash_key='ms' if ps_active else None, bottom_color=AMBER if ms is not None else MUTED)
        self._cell(painter, 540, 340, 40, 65, "DI", str(di) if di is not None else "1b", ps_border,
                   flash_key='di' if ps_active else None, bottom_color=AMBER if di is not None else MUTED)
        self._cell(painter, 580, 340, 80, 65, "PS address", str(ps_addr) if ps_addr is not None else "2 bits",
                   TEAL if ps_active else DIM, fill=TEAL_FILL if ps_active else None,
                   flash_key='ps_addr' if ps_active else None, bottom_color=TEAL if ps_addr is not None else MUTED)

        self._label(painter, 665, 300, 230, "Group type 2 · RadioText", rt_label_color, size=9, bold=True)
        text_ab = v.get('text_ab') if rt_active else None
        rt_addr = v.get('rt_addr') if rt_active else None
        toggle_frac = self._alpha('text_ab_toggle')
        text_ab_border = AMBER if (rt_active and toggle_frac > 0.02) else (AMBER if rt_active else DIM)
        self._cell(painter, 680, 340, 40, 65, "text\nA/B", str(text_ab) if text_ab is not None else "1b",
                   text_ab_border, fill=AMBER_FILL if rt_active else None,
                   flash_key='text_ab' if rt_active else None, bottom_color=AMBER if text_ab is not None else MUTED)
        if toggle_frac > 0.02:
            glow = QColor(AMBER)
            glow.setAlpha(int(220 * toggle_frac))
            pen = QPen(glow, 3)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(QRectF(678, 338, 44, 69))
        self._cell(painter, 720, 340, 160, 65, "RT address", str(rt_addr) if rt_addr is not None else "4 bits",
                   TEAL if rt_active else DIM, fill=TEAL_FILL if rt_active else None,
                   flash_key='rt_addr' if rt_active else None, bottom_color=TEAL if rt_addr is not None else MUTED)

        self._label(painter, 520, 415, 360, "", AMBER, size=9)
        self._label(painter, 520, 432, 360, "", AMBER, size=9)

        painter.end()


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

        diagram_title = QLabel("Live packet anatomy")
        diagram_title_font = diagram_title.font()
        diagram_title_font.setBold(True)
        diagram_title.setFont(diagram_title_font)

        self.diagram = PacketAnatomyWidget()
        self.diagram.setToolTip(
            "The actual A/B/C/D block structure of the RDS group currently "
            "being received, with each field lighting up as its value "
            "resolves. The inactive group-type branch dims out."
        )

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
        self.log.setMaximumBlockCount(500)

        layout = QVBoxLayout()
        layout.addWidget(self.station_label)
        layout.addWidget(self.radiotext_label)
        layout.addWidget(self.status_label)
        layout.addWidget(diagram_title)
        layout.addWidget(self.diagram)
        layout.addWidget(log_title)
        layout.addWidget(self.log)
        self.setLayout(layout)
        self.resize(940, 820)

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
        self.diagram._flash = {}
        self.diagram._values = {}
        self.diagram.update()

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
                    self.diagram.note_group(group)
                else:
                    line = f"[{group['group_type']}{group['version']}] PI={group['pi']:04X} (unhandled group type)"
                self.log.appendPlainText(line)
            self._log_shown = groups.group_count

        self.diagram.update()  # keeps flash fade-out animating even with no new groups

    def closeEvent(self, event):
        self._timer.stop()
        self.worker.stop()
        self.worker.wait(2000)
        event.accept()
