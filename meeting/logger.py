import logging
import json
import os
import sys
from logging.handlers import RotatingFileHandler, SysLogHandler

LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "meeting_cli.log")


class JSONFormatter(logging.Formatter):
    def format(self, record):
        log = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname.lower(),
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "app": "meeting-cli"
        }
        return json.dumps(log, ensure_ascii=False)


def logger_configuration(verbose=False, debug=False):
    os.makedirs(LOG_DIR, exist_ok=True)
    logger = logging.getLogger("meeting-cli")
    logger.setLevel(logging.DEBUG if debug else logging.INFO)

    # --- Formatos ---
    text_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")
    json_formatter = JSONFormatter()

    # --- Consola (texto legible) ---
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(text_formatter if verbose else json_formatter)
    logger.addHandler(ch)

    # --- Archivo rotativo (JSON para Loki/Promtail) ---
    fh = RotatingFileHandler(LOG_FILE, maxBytes=5*1024*1024, backupCount=5)
    fh.setFormatter(json_formatter)
    logger.addHandler(fh)

    # --- Systemd Journal ---
    try:
        syslog_handler = SysLogHandler(address="/dev/log")
        syslog_handler.setFormatter(logging.Formatter("meeting-cli: %(message)s"))
        logger.addHandler(syslog_handler)
    except Exception:
        pass

    return logger
