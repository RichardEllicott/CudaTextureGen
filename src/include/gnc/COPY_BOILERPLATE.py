"""

copy a region of code from a file to another file (overwrite)

ie. anything between

#pragma region REGION
#pragma endregion

gets copied to another file


⚠️
we could change to pathlib
from pathlib import Path

"""
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def extract_region(
    src_path: str,
    dst_path: str,
    region_name: str,
    notes: str = ""
):

    start_marker = f"#pragma region {region_name}"
    end_marker = "#pragma endregion"

    inside = False
    collected = []

    with open(src_path, "r", encoding="utf-8") as f:
        for line in f:
            if start_marker in line:
                inside = True
            if inside:
                collected.append(line)
            if inside and end_marker in line:
                break

    if not collected:
        raise RuntimeError(f"Region '{region_name}' not found in {src_path}")

    # --- overwrite warning ---
    if os.path.exists(dst_path):
        print(f"Warning: {dst_path} already exists.")
        answer = input("Overwrite? [y/N]: ").strip().lower()
        if answer != "y":
            print("Aborted.")
            return

    with open(dst_path, "w", encoding="utf-8") as f:
        f.write(notes)
        f.writelines(collected)
        

    print(f"Extracted region '{region_name}' to {dst_path}")



def generate(dst_filename = "gnc_boilerplate.cuh"):

    src_filename = "gnc_template.cuh"
    
    notes = ""
    notes += "//\n"
    notes += f"// ⚠️ THIS FILE IS COPIED OR GENERATED FROM '{src_filename}'\n"
    notes += "//\n"
    notes += "\n"

    extract_region(
        src_path=os.path.join(SCRIPT_DIR, src_filename),
        dst_path=os.path.join(SCRIPT_DIR, dst_filename),
        region_name="BOILERPLATE",
        notes=notes
    )


# generate("gnc_boilerplate.cuh")
generate("_gnc_boilerplate.cuh")