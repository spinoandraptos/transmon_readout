# --- Config ---

RR = 'rrdummy'

INPUT_FILE = f"/home/spinoandraptos/Documents/CQT/Experiments/Clear Optimisation/{RR}_ClearParam.txt"
OUTPUT_FILE = INPUT_FILE
MAX_AMP_THRESHOLD = 0.65

def parse_value(value):
    value = value.strip().rstrip(',')
    if value.startswith('"') and value.endswith('"'):
        return value.strip('"')
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value

# --- Load parameters from text file ---
pulse_params = {}
with open(INPUT_FILE, 'r') as f:
    for line in f:
        if '=' in line:
            key, val = line.split('=', 1)
            pulse_params[key.strip()] = parse_value(val)

# --- Amplitudes to scale (exclude I_ampx, Q_ampx) ---
amp_keys = [k for k in pulse_params if k.endswith("_amp") and k not in ("I_ampx", "Q_ampx")]

# --- Safely get max amp ---
numeric_amp_values = [abs(pulse_params[k]) for k in amp_keys if isinstance(pulse_params[k], (int, float))]

if numeric_amp_values:
    max_amp = max(numeric_amp_values)

    if max_amp > MAX_AMP_THRESHOLD:
        scale_factor = MAX_AMP_THRESHOLD / max_amp
        print(f"Scaling amplitudes by factor: {scale_factor:.6f}")
        for k in amp_keys:
            if isinstance(pulse_params[k], (int, float)):
                pulse_params[k] *= scale_factor
    else:
        print("No scaling needed.")
else:
    print("No numeric amplitudes found to scale.")

# --- Save updated parameters ---
with open(OUTPUT_FILE, 'w') as f:
    for k, v in pulse_params.items():
        if isinstance(v, str):
            f.write(f'{k} = "{v}",\n')
        else:
            f.write(f'{k} = {v},\n')

print(f"\nScaled pulse parameters written to '{OUTPUT_FILE}'")