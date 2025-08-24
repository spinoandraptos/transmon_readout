class ClearFormatter:
    def __init__(self, **kwargs):

        self.base_amp = 0.2

        # Time fields that are nanoseconds and should be converted to seconds
        self.ns_time_keys = [
            'length', 'pad',
            'ringdown1_time', 'ringup1_time',
            'ringdown2_time', 'ringup2_time',
            'drive_time'
        ]

        self.amp_keys = [
            'ringdown1_amp', 'ringup1_amp',
            'ringdown2_amp', 'ringup2_amp',
            'drive_amp'
        ]

        # First, set all attributes from kwargs (raw values)
        for key, value in kwargs.items():
            setattr(self, key, value)

        # Fix length and pad to ensure consistency
        self._fix_length_and_pad()

        # Convert time fields from ns → s (i.e., float * 1e-9)
        for key in self.ns_time_keys:
            value = getattr(self, key, None)
            if value is not None:
                setattr(self, key, float(value) * 1e-9)

        for key in self.amp_keys:
            value = getattr(self, key, None)
            if value is not None:
                setattr(self, key, float(value) * self.base_amp * self.I_ampx)

    def _fix_length_and_pad(self):
        length_ns = sum(
            getattr(self, k)
            for k in self.ns_time_keys
            if k.endswith('_time')
        )
        self.length = length_ns  # will later be converted to seconds

        pad = getattr(self, 'pad', 0)
        total_ns = self.length + pad
        remainder = total_ns % 64
        if remainder != 0:
            pad += (64 - remainder)
        self.pad = pad  # will be converted to seconds later

    def to_script(self):
        lines = []
        for key in vars(self):
            value = getattr(self, key)
            if key in self.ns_time_keys:
                # Convert seconds to x.xe-9 format
                formatted_value = f"{float(value):.1e}"
            else:
                formatted_value = str(value)
            lines.append(f"{key} = {formatted_value}")
        return "\n".join(lines)
