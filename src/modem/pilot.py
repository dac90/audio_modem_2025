from .constants import DATA_PER_PILOT

def interleave_pilot_blocks(data_blocks, pilot_blocks):
    result = []
    data_index = 0

    for i in range(len(pilot_blocks)):
        # Add a pilot block
        result.append(pilot_blocks[i])
        
        # Add data blocks if it's not the last pilot
        if i < len(pilot_blocks) - 1:
            for _ in range(DATA_PER_PILOT):
                if data_index < len(data_blocks):
                    result.append(data_blocks[data_index])
                    data_index += 1
    return result
