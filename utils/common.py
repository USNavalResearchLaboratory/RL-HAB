def convert_range(value, old_min, old_max, new_min, new_max):
    # Calculate the proportional value
    new_value = new_min + ((value - old_min) / (old_max - old_min)) * (new_max - new_min)
    return new_value

def quarter(x):
    return round(x*4)/4

