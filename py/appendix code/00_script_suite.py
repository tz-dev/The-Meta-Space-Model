# Script: 00_script_suite.py
# Description: Graphical launcher for the Meta-Space Model (MSM) Python simulation suite.
# Features:
#   - GUI with buttons for scripts 01–10, sequentially enabled based on results.csv.
#   - Real-time output, code, and JSON-config display in a scrolled text area.
#   - Buttons for package installation and img folder.
#   - Progress bar below output field for script execution and package installation.
#   - Auto-clears script-related CSV rows before re-execution.
# Dependencies: tkinter, subprocess, threading, csv, json, tabulate, logging

import tkinter as tk
from tkinter import scrolledtext, messagebox, ttk
import subprocess
import threading
import sys, io, os, csv, json, logging
from tabulate import tabulate

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
    "12_meta_test_validator.py"
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
    "12_meta_test_validator.py"
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
    "12_meta_test_validator.py": 17
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
    "12_meta" : "12_meta_test_validator.py"
}

REQUIRED_PACKAGES = sorted(set([
    "numpy", "cupy", "matplotlib", "json", "csv", "logging", "glob", "os", "datetime",
    "tqdm", "scipy", "tkinter", "tabulate", "platform", "astropy"
]))

FONT_SIZE = [10]
CONTENT = {"Output": "", "Code": "", "Config": ""}
CURRENT_VIEW = ["Output"]
IMG_DIR = "img"

def install_missing_packages(output_widget, progress_bar):
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

    threading.Thread(target=task, daemon=True).start()

def run_script(script_name, output_widget, progress_bar, buttons):
    def task():
        try:
            with open("results.csv", "r", encoding="utf-8") as f:
                rows = list(csv.reader(f))
            # Filter out rows for the current script
            filtered = [row for row in rows if script_name not in row[0]]
            filtered.sort(key=sort_key)  # Sort using the global sort_key
            with open("results.csv", "w", newline='', encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerows(filtered)
            logging.info(f"Cleared {script_name} entries from results.csv")
        except FileNotFoundError:
            logging.warning("results.csv not found, creating new")

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

        try:
            process = subprocess.Popen(
                ["python", script_name],
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
                progress_bar['value'] = min((i + 1) * 10, 100)  # Dummy progress
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

        # Update button states after script execution
        for i, btn in enumerate(buttons):
            btn.config(state='normal' if is_script_enabled(i) else 'disabled')

    threading.Thread(target=task, daemon=True).start()

def sort_key(row):
    global SCRIPT_WEIGHTS, ALL_SCRIPTS
    
    if not row or len(row) == 0:
        return (999, '', 999, 0)  # Push invalid to end with default weight
    
    name = row[0].lower()
    try:
        prefix = name.split("_")[0]  # e.g., '10a', '10', '11'
        digits = ''.join(filter(str.isdigit, prefix))  # Extract digits (e.g., '10')
        letters = ''.join(filter(str.isalpha, prefix))  # Extract letters (e.g., 'a' from '10a')
        num = int(digits) if digits else 999
        # Use letters directly for alphabetical sorting, empty string for no subscript
        letter_sort = letters if letters else ''
        # Fallback to ALL_SCRIPTS index only for unknown scripts or duplicates
        sub_idx = ALL_SCRIPTS.index(name) if name in ALL_SCRIPTS else float('inf')
        # Use weight from SCRIPT_WEIGHTS, default to 0 for unknown scripts
        weight = SCRIPT_WEIGHTS.get(name, 0)
        return (num, letter_sort, sub_idx, weight)
    except:
        return (999, '', float('inf'), 0)  # Fallback for errors with default weight

def show_results(output_widget):
    fixed_headers = ['script', 'parameter', 'value', 'target', 'deviation', 'timestamp']
    try:
        # Read results.csv
        try:
            with open("results.csv", "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                data = list(reader)
        except FileNotFoundError:
            # Create results.csv with fixed headers if it doesn't exist
            with open("results.csv", "w", newline='', encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(fixed_headers)
            data = [fixed_headers]
            logging.info("Created results.csv with fixed headers")

        if not data:
            raise ValueError("CSV is empty")
        
        # Validate and clean header
        header = data[0]
        if header != fixed_headers:
            logging.warning(f"Invalid header in results.csv: {header}. Expected: {fixed_headers}")
            data[0] = fixed_headers  # Replace invalid header with fixed_headers
        
        # Filter valid rows and remove empty lines or duplicate headers
        rows = []
        for row in data[1:]:
            if row and len(row) >= len(fixed_headers) and row != fixed_headers:
                rows.append(row)
        logging.info(f"Read {len(data)-1} rows from results.csv, {len(rows)} valid after filtering")
        
        # Sort rows by first column (script name) using the same sort_key
        rows.sort(key=sort_key)
        
        # Generate table without duplicating header in last line
        table = tabulate(rows, headers=fixed_headers, tablefmt="fancy_grid")
        output_widget.configure(state='normal')
        output_widget.delete('1.0', tk.END)
        output_widget.insert(tk.END, "" + table)
        output_widget.configure(state='disabled')
        CONTENT["Output"] = f"Columns: {', '.join(fixed_headers)}\n\n" + table
        CURRENT_VIEW[0] = "Output"
        logging.info(f"Displayed results.csv with columns: {', '.join(fixed_headers)}")
        
    except Exception as e:
        msg = f"Error reading results.csv: {e}"
        output_widget.configure(state='normal')
        output_widget.delete('1.0', tk.END)
        output_widget.insert(tk.END, msg)
        output_widget.configure(state='disabled')
        CONTENT["Output"] = msg
        logging.error(msg)

def reset_results(output_widget, buttons):
    try:
        with open("results.csv", "w", encoding="utf-8") as f:
            pass
        msg = "results.csv has been cleared.\n"
        output_widget.configure(state='normal')
        output_widget.delete('1.0', tk.END)
        output_widget.insert(tk.END, msg)
        output_widget.configure(state='disabled')
        CONTENT["Output"] = msg
        logging.info("Cleared results.csv")
        # Update button states after reset
        for i, btn in enumerate(buttons):
            btn.config(state='normal' if is_script_enabled(i) else 'disabled')
    except Exception as e:
        msg = f"Error clearing results.csv: {e}\n"
        output_widget.configure(state='normal')
        output_widget.delete('1.0', tk.END)
        output_widget.insert(tk.END, msg)
        output_widget.configure(state='disabled')
        CONTENT["Output"] = msg
        logging.error(msg)

def open_img_folder(output_widget):
    try:
        os.makedirs(IMG_DIR, exist_ok=True)
        os.startfile(IMG_DIR)
        logging.info(f"Opened img folder: {IMG_DIR}")
    except Exception as e:
        msg = f"Error opening img folder: {e}\n"
        output_widget.configure(state='normal')
        output_widget.delete('1.0', tk.END)
        output_widget.insert(tk.END, msg)
        output_widget.configure(state='disabled')
        CONTENT["Output"] = msg
        logging.error(msg)

def switch_view(view, output_widget):
    output_widget.configure(state='normal')
    output_widget.delete('1.0', tk.END)
    output_widget.insert(tk.END, CONTENT.get(view, f"No {view.lower()} content available."))
    output_widget.configure(state='disabled')
    CURRENT_VIEW[0] = view
    logging.info(f"Switched view to {view}")

def adjust_font(output_widget, delta):
    FONT_SIZE[0] = max(8, FONT_SIZE[0] + delta)
    output_widget.configure(font=("Consolas", FONT_SIZE[0]))
    logging.info(f"Font size adjusted to {FONT_SIZE[0]}")

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
    # Center the window on the screen
    root.update_idletasks()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    window_width = 1500
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

    # Output field with adjustable width and height (in characters)
    OUTPUT_WIDTH = 80  # Approx. 640px with Consolas 10
    OUTPUT_HEIGHT = 30  # Approx. 450px with Consolas 10
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
        "\n\n\n\n                                     __  __   ____    __  __     ____                  _           _       ____            _   _          \n"
        "                                    |  \\/  | / ___|  |  \\/  |   / ___|    ___   _ __  (_)  _ __   | |_    / ___|   _   _  (_) | |_    ___ \n"
        "                                    | |\\/| | \\___ \\  | |\\/| |   \\___ \\   / __| | '__| | | | '_ \\  | __|   \\___ \\  | | | | | | | __|  / _ \\\n"
        "                                    | |  | |  ___) | | |  | |    ___) | | (__  | |    | | | |_) | | |_     ___) | | |_| | | | | |_  |  __/\n"
        "                                    |_|  |_| |____/  |_|  |_|   |____/   \\___| |_|    |_| | .__/   \\__|   |____/   \\__,_| |_|  \\__|  \\___|\n"
        "                                                                                          |_|                                                \n\n"
        "                                    Welcome to the MSM Script Suite!\n\n"
        "                                    This application provides a graphical interface to run and manage the Meta-Space Model (MSM) simulation scripts.\n\n"
        "                                    FEATURES:\n\n"
        "                                    - Run scripts 01 to 12 sequentially based on results.csv.\n"
        "                                    - View real-time output, code, and JSON configuration in the text area.\n"
        "                                    - Install required packages or open the image folder with dedicated buttons.\n"
        "                                    - Check the progress bar to track script execution and package installation.\n"
        "                                    - Adjust font size with A+ and A- buttons.\n\n"
        "                                    INSTRUCTIONS:\n\n"
        "                                    - Click a script button to run it (enabled ONLY after the previous script succeeds).\n"
        "                                    - Click 'Show results.csv' to check or 'Reset results.csv' to clear calculated values.\n"
        "                                    - Check plotted diagrams by clicking the 'Open Image Folder' button.\n\n"
        "                                    EXTERNAL DATA REQUIREMENTS:\n\n"
        "                                    - For script 10 ('10_external_data_validator.py'), download the SDSS spectral catalog:\n"
        "                                      https://data.sdss.org/sas/dr17/sdss/spectro/redux/specObj-dr17.fits\n"
        "                                      → Place the file in the script directory before execution.\n\n"
        "                                    - For script 11 ('11_2mass_psc_validator.py'), download 2MASS All-Sky PSC archive files from:\n"
        "                                      https://irsa.ipac.caltech.edu/2MASS/download/allsky/\n"
        "                                      → Unpack any number of downloaded tiles into the script directory.\n\n"
        "                                    PLEASE NOTE: '04_empirical_validator.py' can be rerun after scripts 05–11 for validation."
    )
    output_text.configure(state='disabled')

    CONTENT["Output"] = output_text.get('1.0', tk.END)

    progress_bar = ttk.Progressbar(output_frame, length=400, mode='determinate')
    progress_bar.pack(fill=tk.X, pady=(5, 10))

    buttons = []
    for i, script in enumerate(SCRIPTS):
        btn = tk.Button(
            button_frame,
            text=script,
            width=30,
            command=lambda s=script, i=i: is_script_enabled(i) and run_script(s, output_text, progress_bar, buttons),
            state='normal' if is_script_enabled(i) else 'disabled'
        )
        btn.pack(pady=5)
        buttons.append(btn)

    tk.Button(button_frame, text="Show results.csv", width=30,
              command=lambda: show_results(output_text)).pack(pady=(20, 5))
    tk.Button(button_frame, text="Reset results.csv", width=30,
              command=lambda: reset_results(output_text, buttons)).pack(pady=5)
    tk.Button(button_frame, text="Open Image Folder", width=30,
              command=lambda: open_img_folder(output_text)).pack(pady=5)
    tk.Button(button_frame, text="Install Packages", width=30,
              command=lambda: install_missing_packages(output_text, progress_bar)).pack(pady=(20, 5))
    tk.Button(button_frame, text="Exit", width=30, command=root.destroy).pack(pady=(20, 5))

    tk.Button(control_frame, text="Output", command=lambda: switch_view("Output", output_text)).pack(side=tk.LEFT, padx=5)
    tk.Button(control_frame, text="Code", command=lambda: switch_view("Code", output_text)).pack(side=tk.LEFT, padx=5)
    tk.Button(control_frame, text="Config", command=lambda: switch_view("Config", output_text)).pack(side=tk.LEFT, padx=5)
    tk.Label(control_frame, text="PLEASE NOTE: '04_empirical_validator.py' can be rerun after scripts 05–11 for validation.", font=("Arial", 10)).pack(side=tk.LEFT, padx=5)
    tk.Button(control_frame, text="A+", command=lambda: adjust_font(output_text, 1)).pack(side=tk.RIGHT, padx=5)
    tk.Button(control_frame, text="A -", command=lambda: adjust_font(output_text, -1)).pack(side=tk.RIGHT, padx=5)

    root.mainloop()

if __name__ == "__main__":
    create_gui()