import re

RR = 'rrB'

input_file = f"/home/spinoandraptos/Documents/CQT/Experiments/Clear Optimisation/{RR}_ClearParam.txt"
output_file = input_file

# Define amplitude and time keys
amp_keys = {
    'ringdown1_amp',
    'ringup1_amp',
    'ringdown2_amp',
    'ringup2_amp',
    'drive_amp'
}

time_keys = {
    'ringdown1_time',
    'ringup1_time',
    'ringdown2_time',
    'ringup2_time',
    'drive_time',
    'pad'
}

with open(input_file, 'r') as f:
    lines = f.readlines()

# Start with scale definition
scaled_lines = [""]

for line in lines:
    match = re.match(r'\s*(\w+)\s*=\s*([\-0-9\.eE]+)', line)
    if not match:
        continue  # Skip lines that don't match

    key, value = match.group(1), float(match.group(2))

    if key in amp_keys:
        new_line = f"{key} = {value}\n"
    elif key in time_keys:
        new_line = f"{key} = {value}e-9\n"
    else:
        new_line = f"{key} = {value}\n"

    scaled_lines.append(new_line)

with open(output_file, 'w') as f:
    f.writelines(scaled_lines)

print(f"Formatted output saved to {output_file}")