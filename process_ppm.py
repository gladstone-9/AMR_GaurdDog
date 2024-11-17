def analyze_ppm_p6(file_path):
    with open(file_path, 'rb') as file:
        # Read magic number
        magic_number = file.readline().strip()
        if magic_number != b"P6":
            raise ValueError("Unsupported PPM format. This script supports only P6.")

        # Skip comments
        line = file.readline()
        while line.startswith(b'#'):
            line = file.readline()

        # Read dimensions and max color value
        dimensions = line.strip().split()
        width, height = int(dimensions[0]), int(dimensions[1])
        max_color = int(file.readline().strip())
        
        # Check Paramters Read
        print(f'_____Metadata_____\nFormat: {magic_number}\nWidth: {width}\nHeight {height}\nMax Color {max_color}\n')

        # Validate max color value
        if max_color != 255:
            raise ValueError("Unsupported max color value. This script assumes 255.")

        # Read binary pixel data
        pixel_data = file.read()

        # # Analyze pixel data
        # pixels = []
        # idx = 0
        # for i in range(height):
        #     row = []
        #     for j in range(width):
        #         # Each pixel has 3 bytes: R, G, B
        #         r = pixel_data[idx]
        #         g = pixel_data[idx + 1]
        #         b = pixel_data[idx + 2]
        #         row.append((r, g, b))
        #         idx += 3
        #     pixels.append(row)

        # # Print analysis
        # for i, row in enumerate(pixels):
        #     for j, (r, g, b) in enumerate(row):
        #         print(f"Pixel at ({i}, {j}): R={r}, G={g}, B={b}")
        
        # Collect unique RGB values
        unique_colors = set()
        idx = 0
        for _ in range(height):
            for _ in range(width):
                # Each pixel has 3 bytes: R, G, B
                r = pixel_data[idx]
                g = pixel_data[idx + 1]
                b = pixel_data[idx + 2]
                unique_colors.add((r, g, b))
                idx += 3

        # Output all unique RGB values
        print(f"Found {len(unique_colors)} unique RGB values:")
        for rgb in unique_colors:
            print(f"RGB: {rgb}")

# Example Usage
analyze_ppm_p6("openslam_evg-thin/test2_skeleton.ppm")

'''
PPM Pixel Colors:
- RGB: (127, 127, 127): Grey - Blocked off by walls
- RGB: (0, 0, 0): Black - Walls
- RGB: (255, 255, 255): White - Free Space
- RGB: (255, 0, 0): Red - Skeleton Path
'''

# Using: https://github.com/OpenSLAM-org/openslam_evg-thin