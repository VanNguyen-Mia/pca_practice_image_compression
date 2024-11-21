import argparse
import numpy as np
from src.module_1 import (
    create_image_directory,
    display_image, 
    rescale_image, 
    plot_images
)

def main():
    parser = argparse.ArgumentParser(description="Rescale image")
    parser.add_argument('image_path', type=str, help="Path to the image.")
    parser.add_argument('save_path', type=str, help="Path to save rescaled images and covariance plots.")
    parser.add_argument('--save_name', type=str, default="object", help="Base name for saved plots.")
    args = parser.parse_args()

    # Create the plot directory
    create_image_directory(args.save_path)

    # 1. Display image
    image = display_image(args.image_path)

    # 2. Rescale image
    blue_green_red_images, blue, green, red = rescale_image(image, args.save_path, args.save_name)

    # 3. Plot images
    plot_images(blue, green, red, args.save_path, args.save_name)


if __name__ == '__main__':
    main() 