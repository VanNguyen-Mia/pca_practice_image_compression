import numpy as np
import os
from PIL import Image
import cv2
from matplotlib import pyplot as plt

def create_image_directory(save_path: str, subfolder: str = "images") -> str:
    """
    Create directory for saving plot images if they don't exist

    Args:
        save_path (str): path to the parent folder
        subfolder (str): subfolder name for plots

    Returns:
        str: path to the created subfolder
    """
    images_folder = os.path.join(save_path, subfolder)
    os.makedirs(images_folder, exist_ok=True)
    return images_folder

def display_image(image_path: str):
    """
    Read image, display the image and its shape

    Args:
        path (str): Path to the image
    """
    image = Image.open(image_path)

    plt.figure(figsize=[10,6])

    plt.imshow(image);

    print(f"\n Actually Numpy already splits image into three chanels (RGB) red, green and blue: \n\n {np.array(image).shape}")

    return image

def rescale_image(image, save_path: str, save_name: str) -> str:
    """
    Rescale image and separate data into 3 channels: blue, green, red. 

    Args:
        image (_type_): Image from display_image
        save_path (str): Path to save images in blue, green, red
        save_name (str): Name of the images
    
    Returns:
        str: path to the images
    """
    # rescaling image data
    scaled_image = np.array(image) / np.array(image).max()

    # Separate data into 3 channels: blue, green, red.
    blue, green, red = np.array(scaled_image)[:,:,0], np.array(scaled_image)[:,:,1], np.array(scaled_image)[:,:,2]
    print(f"\n blue shape: {blue.shape}, green shape: {green.shape}, red shape: {red.shape}")

    # Create directory for saving images
    images_folder = create_image_directory(save_path)

    # Plot image in 3 channels: blue, green, red
    colors={'blue':blue,'green':green,'red':red}

    fig = plt.figure(figsize=(10, 7))

    for i,ii in zip(colors,range(0,len(colors),1)):
            
        plt.subplot(1,3,ii+1)
        
        plt.imshow(colors[i])
        
        plt.title(str(i))

    blue_green_red_images = os.path.join(images_folder, f"blue_green_red{save_name}.png")
    plt.savefig(blue_green_red_images)

    return blue_green_red_images, blue, green, red

def plot_images(blue: np.array, green: np.array, red: np.array, save_path: str, save_name: str):
    """
    Plot covariance and return indicies of blue, green, red colors in descending order

    Args:
        blue (np.array): np.array of blue from rescale_image function
        green (np.array): np.array of gree from rescale_image function
        red (np.array): np.array of red from rescale_image function
        save_path (str): path to save the covariance plot
        save_name (str): name variable of the covariance plot

    Returns:
        _type_: path to the covariance plot and the indicies of blue, green, red colors
    """
    # Obtain the covariance matrix
    red_cov_mat = np.cov(red.T)
    green_cov_mat = np.cov(green.T)
    blue_cov_mat = np.cov(blue.T)

    # Perform eigendecomposition
    red_eig_vals, red_eig_vecs = np.linalg.eig(red_cov_mat)
    green_eig_vals, green_eig_vecs = np.linalg.eig(green_cov_mat)
    blue_eig_vals, blue_eig_vecs = np.linalg.eig(blue_cov_mat)

    eig_vals={'blue_eig_vals':blue_eig_vals,'green_eig_vals':green_eig_vals,'red_eig_vals':red_eig_vals}

    # Create directory for saving images
    images_folder = create_image_directory(save_path)

    fig = plt.figure(figsize=(16,4))

    for i,ii in zip(eig_vals,range(0,len(eig_vals),1)):
        
        tot = sum(eig_vals[i]) # total variance
        
        var_exp = [(iii / tot) for iii in sorted(eig_vals[i], reverse=True) ] # Individual explained variance
        
        cum_var_exp = np.cumsum(var_exp) # Cumulative explained variance
        
        plt.subplot(1,3,ii+1)
        
        plt.bar(range(1,21),var_exp[:20], alpha = 0.5, label='Variance explained by individual component') # Taking 20 principal components
        
        plt.step(range(1,21),cum_var_exp[:20],where='mid',label='Joint explained variance')
        
        plt.ylabel('Explained variance')
        
        plt.xlabel('Sorted eigen values')
        
        plt.legend(loc=5,prop={'size':8})
        
        plt.title(str(i)[:-9])

    covariance_plot = os.path.join(images_folder, f"covariance_plot{save_name}.png")
    plt.savefig(covariance_plot)

    # Sort eigenvalues in descending order
    red_sorted_indices = np.argsort(red_eig_vals)[::-1]
    green_sorted_indices = np.argsort(green_eig_vals)[::-1]
    blue_sorted_indices = np.argsort(blue_eig_vals)[::-1]

    # Plot the images with number of components
    num_of_pc=[10, 20, 50, 100, 500, 1400, 1800]

    fig = plt.figure(figsize=(20, 20))

    for i,ii in zip(num_of_pc,range(len(num_of_pc))):
        
        # Select the top k eigenvectors
        k = i
        
        red_selected_eig_vecs = red_eig_vecs[:, red_sorted_indices[:i]]
        green_selected_eig_vecs = green_eig_vecs[:, green_sorted_indices[:i]]
        blue_selected_eig_vecs = blue_eig_vecs[:, blue_sorted_indices[:i]]

        # Project the standardized image onto the selected eigenvectors
        red_reduced_image = np.dot(red, red_selected_eig_vecs)
        green_reduced_image = np.dot(green, green_selected_eig_vecs)
        blue_reduced_image = np.dot(blue, blue_selected_eig_vecs)

        # Perform the inverse transformation
        red_reconstructed_image = np.dot(red_reduced_image, red_selected_eig_vecs.T)
        green_reconstructed_image = np.dot(green_reduced_image, green_selected_eig_vecs.T)
        blue_reconstructed_image = np.dot(blue_reduced_image, blue_selected_eig_vecs.T)

        # Combine the color channels
        reconstructed_image = np.dstack((blue_reconstructed_image, green_reconstructed_image,red_reconstructed_image))

        plt.subplot(1,len(num_of_pc),ii+1)
        
        # Display the reconstructed image
        plt.imshow(reconstructed_image)
        
        plt.title(str(i) + ' components')

    resized_pic = os.path.join(images_folder, f"resize_pic{save_name}.png")
    plt.savefig(resized_pic)
        
    return covariance_plot, resized_pic



