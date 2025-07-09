# Script: 12a_render_summary.py
# Description: Renders 12_summary.md in the terminal with ANSI color formatting, parsing Markdown elements manually.
# Author: MSM Enhancement
# Date: 2025-07-09

import os
import sys
import re

# ANSI Farbcode-Konfiguration
STYLES = {
    "H1": "\033[1;96m",   # Hellblau, fett
    "H2": "\033[1;4;97m", # Weiß, fett, unterstrichen
    "H3": "\033[1;93m",   # Gelb, fett
    "BOLD": "\033[1;97m", # Fett für **Text**
    "RESET": "\033[0m",
    "LIST": "  • ",
    "SEPARATOR": "\033[90m" + "─" * 50 + "\033[0m"
}

def render_line(line):
    """Verarbeite eine einzelne Markdown-Zeile mit ANSI-Stil."""
    line = line.rstrip()

    # Header 1
    if line.startswith("# "):
        return f"{STYLES['H1']}{line[2:].strip()}{STYLES['RESET']}"
    
    # Header 2
    if line.startswith("## "):
        return f"{STYLES['H2']}{line[3:].strip()}{STYLES['RESET']}"

    # Header 3
    if line.startswith("### "):
        return f"{STYLES['H3']}{line[4:].strip()}{STYLES['RESET']}"
    
    # Separator
    if re.fullmatch(r"[-=]{3,}", line):
        return STYLES["SEPARATOR"]
    
    # Bullet list
    if line.strip().startswith("- "):
        content = line.strip()[2:].strip()
        return f"{STYLES['LIST']}{replace_bold(content)}"
    
    # Inline bold (z. B. **Wert**)
    return replace_bold(line)

def replace_bold(text):
    """Ersetze **Text** durch fett formatierten Text im Terminal."""
    return re.sub(r"\*\*(.+?)\*\*", rf"{STYLES['BOLD']}\1{STYLES['RESET']}", text)

def render_markdown(file_path="12_summary.md"):
    """Rendere die gesamte Markdown-Datei im Terminal."""
    if not os.path.exists(file_path):
        print(f"[12a] Datei nicht gefunden: {file_path}")
        return

    print(f"{STYLES['H1']}Meta-Space Model (MSM) Summary Viewer{STYLES['RESET']}")
    print(STYLES["SEPARATOR"])
    
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            print(render_line(line))
    
    print(STYLES["SEPARATOR"])

if __name__ == "__main__":
    render_markdown()
