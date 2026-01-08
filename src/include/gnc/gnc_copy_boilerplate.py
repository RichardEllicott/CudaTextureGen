"""


"""
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def extract_region(src_path, dst_path, region_name="BOILERPLATE"):
    start_marker = f"#pragma region {region_name}"
    end_marker   = "#pragma endregion"

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

    with open(dst_path, "w", encoding="utf-8") as f:
        f.writelines(collected)

    print(f"Extracted region '{region_name}' to {dst_path}")



extract_region(
    src_path=os.path.join(SCRIPT_DIR, "gnc_erosion.cuh"),
    dst_path=os.path.join(SCRIPT_DIR, "test_run.cuh"),
    region_name="BOILERPLATE"
)






