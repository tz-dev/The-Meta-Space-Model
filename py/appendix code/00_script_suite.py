# Script: 00_script_suite.py
# Description: Graphical launcher for the Meta-Space Model (MSM) Python simulation suite.
# Features:
#   - GUI with buttons for scripts 01–12, sequentially enabled based on results.csv.
#   - Real-time output, code, and JSON-config display in a scrolled text area.
#   - Buttons for package installation.
#   - Image viewer & Markdown renderer
#   - Progress bar below output field for script execution and package installation.
#   - Auto-clears script-related CSV rows and other outputs (.txt, .log etc) before re-execution.
#   - Mouse wheel scrolling for ImageViewer when active.
# Dependencies: tkinter, subprocess, threading, csv, json, tabulate, logging, PIL

import tkinter as tk
from tkinter import scrolledtext, messagebox, ttk
import subprocess
import threading
import sys, io, os, re, csv, json, logging
from tabulate import tabulate
from PIL import Image, ImageTk

# Logging setup
logging.basicConfig(
    filename='errors.log',
    level=logging.INFO,
    format='%(asctime)s [00_script_suite.py] %(levelname)s: %(message)s'
)

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Globale Definitionen
SCRIPTS = [
    "01_qcd_spectral_field.py",
    "02_monte_carlo_validator.py",
    "03_higgs_spectral_field.py",
    "04_empirical_validator.py",
    "05_s3_spectral_base.py",
    "06_cy3_spectral_base.py",
    "07_gravity_curvature_analysis.py",
    "08_cosmo_entropy_scale.py",
    "09_test_proposal_sim.py",
    "10_external_data_validator.py",
    "11_2mass_psc_validator.py",
    "12_summary.py",
]

ALL_SCRIPTS = [
    "01_qcd_spectral_field.py",
    "02_monte_carlo_validator.py",
    "03_higgs_spectral_field.py",
    "04_empirical_validator.py",
    "05_s3_spectral_base.py",
    "06_cy3_spectral_base.py",
    "07_gravity_curvature_analysis.py",
    "08_cosmo_entropy_scale.py",
    "09_test_proposal_sim.py",
    "10_external_data_validator.py",
    "10a_plot_z_sky_mean.py",
    "10b_neutrino_analysis.py",
    "10c_rg_entropy_flow.py",
    "10d_entropy_map.py",
    "10e_parameter_scan.py",
    "11_2mass_psc_validator.py",
    "12_summary.py",
]

SCRIPT_WEIGHTS = {
    "01_qcd_spectral_field.py": 1,
    "02_monte_carlo_validator.py": 2,
    "03_higgs_spectral_field.py": 3,
    "04_empirical_validator.py": 4,
    "05_s3_spectral_base.py": 5,
    "06_cy3_spectral_base.py": 6,
    "07_gravity_curvature_analysis.py": 7,
    "08_cosmo_entropy_scale.py": 8,
    "09_test_proposal_sim.py": 9,
    "10_external_data_validator.py": 10,
    "10a_plot_z_sky_mean.py": 11,
    "10b_neutrino_analysis.py": 12,
    "10c_rg_entropy_flow.py": 13,
    "10d_entropy_map.py": 14,
    "10e_parameter_scan.py": 15,
    "11_2mass_psc_validator.py": 16,
    "12_summary.py": 17
}

CONFIG_MAP = {
    "01_qcd": "config_qcd.json",
    "02_monte": "config_monte_carlo.json",
    "03_higgs": "config_higgs.json",
    "04_empirical": "config_empirical.json",
    "05_s3": "config_s3.json",
    "06_cy3": "config_cy3.json",
    "07_grav": "config_grav.json",
    "08_cosmo": "config_cosmo.json",
    "09_test": "config_test.json",
    "10_external": "config_external.json",
    "11_2mass": "config_2mass.json",
}

REQUIRED_PACKAGES = sorted(set([
    "numpy", "cupy", "matplotlib", "json", "csv", "logging", "glob", "os", "datetime",
    "tqdm", "scipy", "tkinter", "tabulate", "platform", "astropy", "pillow",
    "subprocess", "sys", "threading", "time", "warnings", "pandas"
]))

FONT_SIZE = [12]
CONTENT = {"Output": "", "Code": "", "Config": ""}
CURRENT_VIEW = ["Output"]
IMG_DIR = "img"
IMG_WIDTH = 800
IMG_HEIGHT = 600
IMG_LIST_MARGIN = 30

def install_missing_packages(output_widget, progress_bar, viewer):
    def task():
        progress_bar['value'] = 0
        output_widget.configure(state='normal')
        output_widget.delete('1.0', tk.END)
        output_widget.insert(tk.END, "Checking and installing required packages...\n\n")
        output_widget.configure(state='disabled')
        for i, pkg in enumerate(REQUIRED_PACKAGES):
            try:
                __import__(pkg)
                output_widget.configure(state='normal')
                output_widget.insert(tk.END, f"✔ {pkg} is already installed\n")
                output_widget.configure(state='disabled')
                logging.info(f"Package {pkg} already installed")
            except ImportError:
                output_widget.configure(state='normal')
                output_widget.insert(tk.END, f"➤ Installing {pkg}...\n")
                output_widget.update()
                output_widget.configure(state='disabled')
                try:
                    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
                    output_widget.configure(state='normal')
                    output_widget.insert(tk.END, f"✓ Successfully installed {pkg}\n")
                    output_widget.configure(state='disabled')
                    logging.info(f"Successfully installed {pkg}")
                except Exception as e:
                    output_widget.configure(state='normal')
                    output_widget.insert(tk.END, f"✗ Failed to install {pkg}: {e}\n")
                    output_widget.configure(state='disabled')
                    logging.error(f"Failed to install {pkg}: {e}")
            progress_bar['value'] = ((i + 1) / len(REQUIRED_PACKAGES)) * 100
        output_widget.configure(state='normal')
        output_widget.insert(tk.END, "\nDone.\n")
        output_widget.configure(state='disabled')
        logging.info("Package installation completed")
        progress_bar['value'] = 100
        viewer.disable_mousewheel()

    threading.Thread(target=task, daemon=True).start()

def run_script(script_name, output_widget, progress_bar, buttons, viewer):
    def task():
        output_widget.configure(state='normal')
        output_widget.delete('1.0', tk.END)
        output_widget.insert(tk.END, f"Running: {script_name}\n\n")
        output_widget.see(tk.END)
        output_widget.configure(state='disabled')
        CURRENT_VIEW[0] = "Output"
        CONTENT["Output"] = f"Running: {script_name}\n\n"
        CONTENT["Code"] = ""
        CONTENT["Config"] = ""

        progress_bar['value'] = 0
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"

        # Multi-Klassen-Unterstützung für 10b und 10d
        cmd = ["python", script_name]
        if script_name in ["10b_neutrino_analysis.py", "10d_entropy_map.py"]:
            config = json.load(open("config_external.json", "r", encoding="utf-8"))
            classes = config.get("enabled_classes", ["GALAXY", "QSO"])
            for cls in classes:
                input_csv = f"z_sky_mean_{cls.lower()}.csv"
                if os.path.exists(input_csv):
                    cmd = ["python", script_name, input_csv]
                    output_widget.configure(state='normal')
                    output_widget.insert(tk.END, f"Running {script_name} for class {cls}...\n")
                    output_widget.see(tk.END)
                    output_widget.configure(state='disabled')
                    CONTENT["Output"] += f"Running {script_name} for class {cls}...\n"
                    try:
                        process = subprocess.Popen(
                            cmd,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            encoding="utf-8",
                            env=env
                        )
                        output = ""
                        for i, line in enumerate(process.stdout):
                            output += line
                            output_widget.configure(state='normal')
                            output_widget.insert(tk.END, line)
                            output_widget.see(tk.END)
                            output_widget.configure(state='disabled')
                            progress_bar['value'] = min((i + 1) * 10, 100)
                            CONTENT["Output"] = output
                        process.stdout.close()
                        process.wait()
                        logging.info(f"Completed {script_name} for class {cls}")
                    except Exception as e:
                        logging.error(f"Failed to run {script_name} for class {cls}: {e}")
                        output_widget.configure(state='normal')
                        output_widget.insert(tk.END, f"Error: {e}\n")
                        output_widget.configure(state='disabled')
                        CONTENT["Output"] = f"Error: {e}\n"
        else:
            try:
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    encoding="utf-8",
                    env=env
                )
                output = ""
                for i, line in enumerate(process.stdout):
                    output += line
                    output_widget.configure(state='normal')
                    output_widget.insert(tk.END, line)
                    output_widget.see(tk.END)
                    output_widget.configure(state='disabled')
                    progress_bar['value'] = min((i + 1) * 10, 100)
                    CONTENT["Output"] = output
                process.stdout.close()
                process.wait()
                logging.info(f"Completed {script_name}")
            except Exception as e:
                logging.error(f"Failed to run {script_name}: {e}")
                output_widget.configure(state='normal')
                output_widget.insert(tk.END, f"Error: {e}\n")
                output_widget.configure(state='disabled')
                CONTENT["Output"] = f"Error: {e}\n"

        try:
            with open(script_name, "r", encoding="utf-8") as f:
                CONTENT["Code"] = f.read()
        except Exception as e:
            CONTENT["Code"] = f"Could not read {script_name}: {e}"
            logging.error(f"Failed to read {script_name}: {e}")

        key = "_".join(script_name.split("_")[:2])
        config_file = CONFIG_MAP.get(key, "")
        if config_file and os.path.exists(config_file):
            try:
                with open(config_file, "r", encoding="utf-8") as f:
                    CONTENT["Config"] = json.dumps(json.load(f), indent=2)
            except Exception as e:
                CONTENT["Config"] = f"Could not read {config_file}: {e}"
                logging.error(f"Failed to read {config_file}: {e}")
        else:
            CONTENT["Config"] = "No config found for this script."

        progress_bar['value'] = 100
        for i, btn in enumerate(buttons):
            btn.config(state='normal' if is_script_enabled(i) else 'disabled')

        # Nach Erstellung von 12_summary.md automatisch MarkdownViewer aufrufen
        if script_name == "12_summary.py" and os.path.exists("12_summary.md"):
            try:
                MarkdownViewer(output_widget).show("12_summary.md")
            except Exception as e:
                output_widget.configure(state='normal')
                output_widget.insert(tk.END, f"\nCould not render Markdown: {e}\n")
                output_widget.configure(state='disabled')

        viewer.disable_mousewheel()

    threading.Thread(target=task, daemon=True).start()

def sort_key(row):
    if not row or len(row) == 0:
        return (999, '', 999, 0)
    name = row[0].lower()
    try:
        prefix = name.split("_")[0]
        digits = ''.join(filter(str.isdigit, prefix))
        letters = ''.join(filter(str.isalpha, prefix))
        num = int(digits) if digits else 999
        letter_sort = letters if letters else ''
        sub_idx = ALL_SCRIPTS.index(name) if name in ALL_SCRIPTS else float('inf')
        weight = SCRIPT_WEIGHTS.get(name, 0)
        return (num, letter_sort, sub_idx, weight)
    except:
        return (999, '', float('inf'), 0)

def show_results(output_widget, viewer):
    fixed_headers = ['script', 'parameter', 'value', 'target', 'deviation', 'timestamp']
    try:
        try:
            with open("results.csv", "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                data = list(reader)
        except FileNotFoundError:
            with open("results.csv", "w", newline='', encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(fixed_headers)
            data = [fixed_headers]
            logging.info("Created results.csv with fixed headers")

        if not data:
            raise ValueError("CSV is empty")
        
        header = data[0]
        if header != fixed_headers:
            logging.warning(f"Invalid header in results.csv: {header}. Expected: {fixed_headers}")
            data[0] = fixed_headers
        
        rows = [row for row in data[1:] if row and len(row) >= len(fixed_headers) and row != fixed_headers]
        logging.info(f"Read {len(data)-1} rows from results.csv, {len(rows)} valid after filtering")
        rows.sort(key=sort_key)
        
        table = tabulate(rows, headers=fixed_headers, tablefmt="fancy_grid")
        output_widget.configure(state='normal')
        output_widget.delete('1.0', tk.END)
        output_widget.insert(tk.END, table)
        output_widget.configure(state='disabled')
        CONTENT["Output"] = f"Columns: {', '.join(fixed_headers)}\n\n" + table
        CURRENT_VIEW[0] = "Output"
        logging.info(f"Displayed results.csv with columns: {', '.join(fixed_headers)}")
        viewer.disable_mousewheel()

    except Exception as e:
        msg = f"Error reading results.csv: {e}"
        output_widget.configure(state='normal')
        output_widget.delete('1.0', tk.END)
        output_widget.insert(tk.END, msg)
        output_widget.configure(state='disabled')
        CONTENT["Output"] = msg
        logging.error(msg)

def reset_results(output_widget, buttons, viewer):
    try:
        # Clear results.csv
        with open("results.csv", "w", encoding="utf-8") as f:
            pass
        # Delete all .csv, .txt, .md, and .log files in base directory and .png in img/ directory
        img_dir = "img"
        if os.path.exists(img_dir):
            for file in os.listdir(img_dir):
                if file.lower().endswith(".png"):
                    try:
                        os.remove(os.path.join(img_dir, file))
                        logging.info(f"Deleted {file} from {img_dir}")
                    except Exception as e:
                        logging.error(f"Failed to delete {file}: {e}")
        # Close logging handlers before deleting errors.log
        for handler in logging.root.handlers[:]:
            handler.flush()
            handler.close()
            logging.root.removeHandler(handler)

        # Delete .csv, .txt, and .log files in base directory
        for file in os.listdir("."):
            if file.lower().endswith((".csv", ".txt", ".md", ".log")):
                try:
                    os.remove(file)
                    # print instead of logging (logging is now inactive)
                    print(f"Deleted {file} from base directory")
                except Exception as e:
                    print(f"Failed to delete {file}: {e}")

        msg = "All results and generated files of these types have been cleared: \n\n  - .csv\n  - .log\n  - .txt\n  - .png"

        output_widget.configure(state='normal')
        output_widget.delete('1.0', tk.END)
        output_widget.insert(tk.END, msg)
        output_widget.configure(state='disabled')
        CONTENT["Output"] = msg
        logging.info("Cleared results.csv, all .csv, .txt, .log files in base directory, and all .png files in img/")
        for i, btn in enumerate(buttons):
            btn.config(state='normal' if is_script_enabled(i) else 'disabled')
        viewer.disable_mousewheel()
    except Exception as e:
        msg = f"Error clearing results.csv, .csv, .txt, .log files, or img/*.png: {e}\n"
        output_widget.configure(state='normal')
        output_widget.delete('1.0', tk.END)
        output_widget.insert(tk.END, msg)
        output_widget.configure(state='disabled')
        CONTENT["Output"] = msg
        logging.error(msg)

class ImageViewer:
    def __init__(self, output_widget, img_dir="img", parent=None):
        self.output_widget = output_widget
        self.img_dir = img_dir
        self.images = sorted([f for f in os.listdir(img_dir) if f.lower().endswith((".png", ".gif"))])
        self.index = 0
        self.tkimages = []
        self.parent = parent
        self.is_active = False

    def display_image(self):
        self.output_widget.configure(state='normal')
        self.output_widget.delete('1.0', tk.END)
        self.is_active = True
        CURRENT_VIEW[0] = "ImageViewer"

        self.images = sorted([f for f in os.listdir(self.img_dir) if f.lower().endswith((".png", ".gif"))])
        if not self.images:
            self._display_text("No images found in img/")
            return

        img_path = os.path.join(self.img_dir, self.images[self.index])
        try:
            img = Image.open(img_path)
            img.thumbnail((IMG_WIDTH, IMG_HEIGHT), Image.LANCZOS)
            tk_img = ImageTk.PhotoImage(img)
            self.tkimages = [tk_img]

            container = tk.Frame(self.output_widget, bg="black")
            img_label = tk.Label(container, image=tk_img, bg="black", width=IMG_WIDTH, height=IMG_HEIGHT)
            img_label.pack(side="left", anchor="nw", padx=0, pady=0)

            # Split images into two columns if more than 10 images
            max_per_column = 21
            first_column = self.images[:max_per_column]
            second_column = self.images[max_per_column:]

            # First column
            list_frame1 = tk.Frame(container, bg="black")
            for i, name in enumerate(first_column):
                fg = "#00ff00" if i == self.index else "#888888"
                lbl = tk.Label(list_frame1, text=name, fg=fg, bg="black", font=("Consolas", 10), anchor="w")
                lbl.pack(anchor="w")
            list_frame1.pack(side="left", anchor="n", padx=IMG_LIST_MARGIN)

            # Second column (if overflow exists)
            if second_column:
                list_frame2 = tk.Frame(container, bg="black")
                for i, name in enumerate(second_column, start=max_per_column):
                    fg = "#00ff00" if i == self.index else "#888888"
                    lbl = tk.Label(list_frame2, text=name, fg=fg, bg="black", font=("Consolas", 10), anchor="w")
                    lbl.pack(anchor="w")
                list_frame2.pack(side="left", anchor="n", padx=IMG_LIST_MARGIN)

            self.output_widget.window_create(tk.END, window=container)
        except Exception as e:
            self._display_text(f"Error loading image: {e}")

    def show(self):
        self.images = sorted([f for f in os.listdir(self.img_dir) if f.lower().endswith((".png", ".gif"))])
        self.index = 0

        if not self.images:
            self._display_text("No images found in img/")
            return

        self.display_image()

        if self.parent:
            self.parent.bind("<MouseWheel>", self._on_mousewheel)
            self.parent.bind("<Left>", self.show_prev)
            self.parent.bind("<Right>", self.show_next)

    def disable_mousewheel(self):
        if self.parent:
            self.parent.unbind("<MouseWheel>")
            self.parent.unbind("<Left>")
            self.parent.unbind("<Right>")
        self.is_active = False

    def show_prev(self, event=None):
        if not self.images:
            return
        self.index = (self.index - 1) % len(self.images)
        self.display_image()

    def show_next(self, event=None):
        if not self.images:
            return
        self.index = (self.index + 1) % len(self.images)
        self.display_image()

    def _on_mousewheel(self, event):
        if self.is_active:
            if event.delta > 0:
                self.show_prev()
            else:
                self.show_next()

    def _display_text(self, msg):
        self.output_widget.configure(state='normal')
        self.output_widget.delete('1.0', tk.END)
        self.output_widget.insert(tk.END, msg)
        self.output_widget.configure(state='disabled')
        self.is_active = False

class MarkdownViewer:
    def __init__(self, output_widget, base_size=12):
        self.output_widget = output_widget
        self.base_size = base_size
        self.setup_tags()

    def setup_tags(self):
        """Configure text styles for Markdown elements (dynamisch mit base_size)."""
        bs = self.base_size  # alias
        self.output_widget.tag_configure("h1", font=("Arial", bs + 6, "bold", "underline"), foreground="#d35400", spacing3=6)
        self.output_widget.tag_configure("h2", font=("Arial", bs + 4, "bold", "underline"), foreground="#5dade2", spacing3=4)
        self.output_widget.tag_configure("h3", font=("Arial", bs + 2, "bold"), foreground="#f1c40f", spacing3=2)
        self.output_widget.tag_configure("bold", font=("Consolas", bs + 2, "bold"))
        self.output_widget.tag_configure("italic", font=("Consolas", bs + 2, "italic"), foreground="#cccccc")
        self.output_widget.tag_configure("bullet", lmargin1=30, lmargin2=50, spacing3=2)
        self.output_widget.tag_configure("codeblock", font=("Courier", bs + 2), background="#1e1e1e", foreground="#eeeeee", lmargin1=20, spacing3=2)
        self.output_widget.tag_configure("inlinecode", font=("Courier", bs + 1), background="#333333", foreground="#eeeeee")
        self.output_widget.tag_configure("separator", foreground="#66ff99")
        self.output_widget.tag_configure("status_pass", font=("Consolas", bs), foreground="#00cc66")
        self.output_widget.tag_configure("status_fail", font=("Consolas", bs), foreground="#ff4d4d")
        self.output_widget.tag_configure("status_na",   font=("Consolas", bs), foreground="#aaaaaa")

    def show(self, filepath):
        """Display and render a markdown file in the output widget."""
        if not os.path.exists(filepath):
            self.output_widget.configure(state="normal")
            self.output_widget.delete("1.0", tk.END)
            self.output_widget.insert(tk.END, f"File not found: {filepath}")
            self.output_widget.configure(state="disabled")
            return

        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()

        self.output_widget.configure(state="normal")
        self.output_widget.delete("1.0", tk.END)

        in_code_block = False

        for line in lines:
            line = line.rstrip("\n")

            # Code block toggle
            if line.strip().startswith("```"):
                in_code_block = not in_code_block
                continue

            if in_code_block:
                self.output_widget.insert(tk.END, line + "\n", "codeblock")
                continue

            # Horizontal rule
            if re.match(r"^-{3,}$", line.strip()):
                self.output_widget.insert(tk.END, "─" * 80 + "\n", "separator")
                continue

            # Headers
            if line.startswith("# "):
                self.output_widget.insert(tk.END, line[2:] + "\n\n", "h1")
                continue

            elif line.startswith("## "):
                self.output_widget.insert(tk.END, "\n" + line[3:] + "\n\n", "h2")
                continue

            elif line.startswith("### "):
                self.output_widget.insert(tk.END, line[4:] + "\n", "h3")
                continue

            # List items
            if line.strip().startswith("- ") or line.strip().startswith("• "):
                content = re.sub(r"^[-•]\s+", "", line.strip())

                # Split in **bold** parts
                parts = re.split(r"(\*\*.*?\*\*)", content)
                self.output_widget.insert(tk.END, "• ", "bullet")
                for part in parts:
                    if part.startswith("**") and part.endswith("**"):
                        self.output_widget.insert(tk.END, part[2:-2], ("bullet", "bold"))
                    else:
                        self.output_widget.insert(tk.END, part, "bullet")
                self.output_widget.insert(tk.END, "\n", "bullet")
                continue

            # Inline code (e.g., `value`)
            def replace_inline_code(match):
                code = match.group(1)
                self.output_widget.insert(tk.END, code, "inlinecode")
                return ""

            if "`" in line:
                parts = re.split(r"`(.*?)`", line)
                for i, part in enumerate(parts):
                    if i % 2 == 0:
                        self.output_widget.insert(tk.END, part)
                    else:
                        self.output_widget.insert(tk.END, part, "inlinecode")
                self.output_widget.insert(tk.END, "\n")
                continue

            # **PASS✅**", "- **FAIL❌**", "- **N/A➖**"
            status_match = re.match(r"^\s*[-•]\s+\*\*(PASS✅|FAIL❌|N/A➖)\*\*", line)
            if status_match:
                status_token = status_match.group(1)
                tag = "status_pass" if "PASS" in status_token else "status_fail" if "FAIL" in status_token else "status_na"
                self.output_widget.insert(tk.END, "• ", "bullet")
                self.output_widget.insert(tk.END, status_token, tag)
                self.output_widget.insert(tk.END, "\n", "bullet")
                continue

            # Bold (**text**) – rendered after inline code
            bold_matches = list(re.finditer(r"\*\*(.*?)\*\*", line))
            if bold_matches:
                last_index = 0
                for m in bold_matches:
                    self.output_widget.insert(tk.END, line[last_index:m.start()])
                    self.output_widget.insert(tk.END, m.group(1), "bold")
                    last_index = m.end()
                self.output_widget.insert(tk.END, line[last_index:] + "\n")
                continue

            # Default: plain text
            self.output_widget.insert(tk.END, line + "\n")

        self.output_widget.configure(state="disabled")


def switch_view(view, output_widget, viewer):
    output_widget.configure(state='normal')
    output_widget.delete('1.0', tk.END)
    output_widget.insert(tk.END, CONTENT.get(view, f"No {view.lower()} content available."))
    output_widget.configure(state='disabled')
    CURRENT_VIEW[0] = view
    logging.info(f"Switched view to {view}")
    viewer.disable_mousewheel()

def adjust_font(output_widget, delta, viewer):
    FONT_SIZE[0] = max(8, FONT_SIZE[0] + delta)
    output_widget.configure(font=("Consolas", FONT_SIZE[0]))
    logging.info(f"Font size adjusted to {FONT_SIZE[0]}")
    viewer.disable_mousewheel()

def is_script_enabled(index):
    if index == 0:
        return True
    prev_script = SCRIPTS[index - 1]
    try:
        with open("results.csv", "r", encoding="utf-8") as f:
            return any(prev_script in row[0] for row in csv.reader(f))
    except:
        return False

def create_gui():
    root = tk.Tk()
    root.title("MSM Script Suite")
    root.update_idletasks()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    window_width = 1850
    window_height = 800
    position_x = (screen_width // 2) - (window_width // 2)
    position_y = (screen_height // 2) - (window_height // 2)
    root.geometry(f"{window_width}x{window_height}+{position_x}+{position_y}")

    button_frame = tk.Frame(root)
    button_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

    output_frame = tk.Frame(root)
    output_frame.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH, padx=10, pady=10)

    control_frame = tk.Frame(output_frame)
    control_frame.pack(fill=tk.X, pady=(0, 10))

    OUTPUT_WIDTH = 80
    OUTPUT_HEIGHT = 30
    output_text = scrolledtext.ScrolledText(
        output_frame,
        wrap=tk.WORD,
        state='disabled',
        font=("Consolas", FONT_SIZE[0]),
        bg="black",
        fg="#00ff00",
        width=OUTPUT_WIDTH,
        height=OUTPUT_HEIGHT
    )
    output_text.pack(fill=tk.BOTH, expand=True)

    output_text.configure(state='normal')
    output_text.insert(tk.END,
        "                                     __  __   ____    __  __     ____                  _           _       ____            _   _          \n"
        "                                    |  \\/  | / ___|  |  \\/  |   / ___|    ___   _ __  (_)  _ __   | |_    / ___|   _   _  (_) | |_    ___ \n"
        "                                    | |\\/| | \\___ \\  | |\\/| |   \\___ \\   / __| | '__| | | | '_ \\  | __|   \\___ \\  | | | | | | | __|  / _ \\\n"
        "                                    | |  | |  ___) | | |  | |    ___) | | (__  | |    | | | |_) | | |_     ___) | | |_| | | | | |_  |  __/\n"
        "                                    |_|  |_| |____/  |_|  |_|   |____/   \\___| |_|    |_| | .__/   \\__|   |____/   \\__,_| |_|  \\__|  \\___|\n"
        "                                                                                          |_|                                                \n"
        "                                Welcome to the MSM Script Suite!\n\n"
        "                                This application provides a graphical interface to run and manage the Meta-Space Model (MSM) simulation scripts.\n\n"
        "                                FEATURES:\n\n"
        "                                - Run scripts 01 to 12 sequentially based on results.csv.\n"
        "                                - View real-time output, code, and JSON configuration in the text area.\n"
        "                                - Install required packages or open the image viewer with dedicated buttons.\n"
        "                                - Check the progress bar to track script execution and package installation.\n"
        "                                - Adjust font size with A+ and A- buttons.\n"
        "                                - Use mouse wheel to scroll images in the Image Viewer.\n\n"
        "                                INSTRUCTIONS:\n\n"
        "                                - Click a script button to run it (enabled ONLY after the previous script succeeds).\n"
        "                                - Click 'Show results.csv' to check or 'Reset results' to clear calculated values, plots etc.\n"
        "                                - View plotted diagrams by clicking the 'Open Image Viewer' button.\n\n"
        "                                EXTERNAL DATA REQUIREMENTS:\n\n"
        "                                - For script 10 ('10_external_data_validator.py'), download the SDSS spectral catalog:\n"
        "                                  https://data.sdss.org/sas/dr17/sdss/spectro/redux/specObj-dr17.fits\n"
        "                                  → Place the file in the script directory before execution.\n\n"
        "                                - For script 11 ('11_2mass_psc_validator.py'), download 2MASS All-Sky PSC archive files from:\n"
        "                                  https://irsa.ipac.caltech.edu/2MASS/download/allsky/\n"
        "                                  → Unpack any number of downloaded tiles into the script directory.\n\n"
        "                                PLEASE NOTE: '04_empirical_validator.py' can be rerun after scripts 05–11 for validation."
    )
    output_text.configure(state='disabled')
    CONTENT["Output"] = output_text.get('1.0', tk.END)

    progress_bar = ttk.Progressbar(output_frame, length=400, mode='determinate')
    progress_bar.pack(fill=tk.X, pady=(5, 10))

    viewer = ImageViewer(output_text, parent=root)
    buttons = []
    for i, script in enumerate(SCRIPTS):
        btn = tk.Button(
            button_frame,
            text=script,
            width=30,
            command=lambda s=script, i=i: is_script_enabled(i) and run_script(s, output_text, progress_bar, buttons, viewer),
            state='normal' if is_script_enabled(i) else 'disabled'
        )
        btn.pack(pady=5)
        buttons.append(btn)

    tk.Button(button_frame, text="Show results.csv", width=30,
              command=lambda: show_results(output_text, viewer)).pack(pady=(20, 5))
    tk.Button(button_frame, text="Reset all results", width=30,
              command=lambda: reset_results(output_text, buttons, viewer)).pack(pady=5)
    tk.Button(button_frame, text="Open Image Viewer", width=30,
              command=viewer.show).pack(pady=5)
    tk.Button(button_frame, text="Install Packages", width=30,
              command=lambda: install_missing_packages(output_text, progress_bar, viewer)).pack(pady=(20, 5))
    tk.Button(button_frame, text="Exit", width=30, command=root.destroy).pack(pady=(20, 5))

    tk.Button(control_frame, text="Output", command=lambda: switch_view("Output", output_text, viewer)).pack(side=tk.LEFT, padx=5)
    tk.Button(control_frame, text="Code", command=lambda: switch_view("Code", output_text, viewer)).pack(side=tk.LEFT, padx=5)
    tk.Button(control_frame, text="Config", command=lambda: switch_view("Config", output_text, viewer)).pack(side=tk.LEFT, padx=5)
    tk.Label(control_frame, text="PLEASE NOTE: '04_empirical_validator.py' can be rerun after scripts 05–11 for validation.", font=("Arial", 10)).pack(side=tk.LEFT, padx=5)
    tk.Button(control_frame, text="A+", command=lambda: adjust_font(output_text, 1, viewer)).pack(side=tk.RIGHT, padx=5)
    tk.Button(control_frame, text="A -", command=lambda: adjust_font(output_text, -1, viewer)).pack(side=tk.RIGHT, padx=5)

    root.mainloop()

if __name__ == "__main__":
    create_gui()
