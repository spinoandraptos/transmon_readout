import re
from pathlib import Path

RR = 'rrC'

# === USER CONFIGURATION ===
MAX = 0.5
MIN = -0.5

MULTIPLY_AMPS = False
MULTIPLY_TIMES = False     # Set True to multiply all *_time fields

INPUT_FILE = str(Path.cwd()) + f"\\scripts\\CLEAR\\{RR}_ClearParam.txt"
OUTPUT_FILE = INPUT_FILE

TARGET_TOTAL_TIME = 50 * 64         # Your desired pad + length total time
TIME_MULTIPLIER = 2     # The factor to multiply times by
AMP_MULTIPLIER = 1.5     # The factor to multiply amps by

# ==========================

def parse_params(text):
    pattern = r"(\w+)\s*=\s*(.*?)(?:,|\n|$)"
    return dict(re.findall(pattern, text))

def format_value(val):
    try:
        fval = float(val)
        if fval.is_integer():
            return str(int(fval))
        return f"{fval:.16g}"
    except ValueError:
        return val.strip()

def modify_params(params, target_total_time=None, multiply_times=False, multiply_amps=False, time_multiplier=1.0, amp_multiplier=1.0):
    changes = []

    # Extract *_time fields
    time_keys = [k for k in params if k.endswith("_time")]
    amp_keys = [k for k in params if k.endswith("_amp")]
    time_values = []
    amp_values = []

    for k in time_keys:
        try:
            val = float(params[k])
        except ValueError:
            raise ValueError(f"Invalid number for time field: {k} = {params[k]}")

        if multiply_times:
            new_val = round(val * time_multiplier)
            changes.append(f"Multiplied {k}: {val} → {new_val}")
            params[k] = str(new_val)
            time_values.append(new_val)
        else:
            time_values.append(val)

    for k in amp_keys:
        try:
            val = float(params[k])
        except ValueError:
            raise ValueError(f"Invalid number for time field: {k} = {params[k]}")
    
        if multiply_amps:
            new_val = val * amp_multiplier
            if new_val > MAX:
                new_val = MAX
            elif new_val < MIN:
                new_val = MIN
            changes.append(f"Multiplied {k}: {val} → {new_val}")
            params[k] = str(new_val)
            amp_values.append(new_val)
        else:
            amp_values.append(val)

    # Recompute length
    length = int(round(sum(time_values)))
    old_length = params.get("length")
    params["length"] = str(length)
    if str(old_length) != str(length):
        changes.append(f"Corrected length: {old_length} → {length}")

    # Recompute pad
    if target_total_time is not None:
        pad = target_total_time - length
        pad_reason = f"target_total_time={target_total_time}"
    else:
        min_time = int(round(min(time_values)))
        pad = min(0, -min_time)
        pad_reason = f"min_time={min_time}"

    old_pad = params.get("pad")
    if str(old_pad) != str(pad):
        changes.append(f"Corrected pad: {old_pad} → {pad} ({pad_reason})")
    params["pad"] = str(pad)

    return params, changes

def write_params(params):
    lines = []
    for k, v in params.items():
        lines.append(f"\t\t\t\t{k} = {format_value(v)},")  # 3 tabs instead of 4 spaces
    return "\n".join(lines) + "\n"

def main():
    with open(INPUT_FILE, "r") as f:
        original_text = f.read()

    params = parse_params(original_text)

    updated_params, change_log = modify_params(
        params,
        target_total_time=TARGET_TOTAL_TIME,
        multiply_times=MULTIPLY_TIMES,
        multiply_amps=MULTIPLY_AMPS,
        time_multiplier=TIME_MULTIPLIER,
        amp_multiplier=AMP_MULTIPLIER
    )

    new_text = write_params(updated_params)

    with open(OUTPUT_FILE, "w") as f:
        f.write(new_text)

    print(f"✔ Saved corrected parameters to '{OUTPUT_FILE}':")
    for change in change_log:
        print("   -", change)

if __name__ == "__main__":
    main()