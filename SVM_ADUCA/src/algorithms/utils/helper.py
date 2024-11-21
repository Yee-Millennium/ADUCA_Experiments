def construct_block_range(dimension, block_size):
    blocks = []

    for start in range(0, dimension, block_size):
        end = min(start + block_size, dimension)  # Ensure the end doesn't exceed total columns
        block_range = range(start, end)
        blocks.append(block_range)
    return blocks