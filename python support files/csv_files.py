# import os
# import csv
# from PIL import Image

# # Set this to 'train', 'test', or 'validate' depending on which set you want to process
# selected_set_type = 'train'

# # Base directory where the 'test', 'train', and 'validate' directories are located
# base_dir = 'OIDv/OID/Dataset'

# # Directory for the specific set you want to process
# set_dir = os.path.join(base_dir, selected_set_type)

# # Output CSV file path
# output_csv_file = f'normalized_annotations_{selected_set_type}.csv'

# # Function to normalize bounding box coordinates
# def normalize_bounding_box(xmin, ymin, xmax, ymax, image_width, image_height):
#     return xmin / image_width, ymin / image_height, xmax / image_width, ymax / image_height

# # Open the CSV file for writing
# with open(output_csv_file, 'w', newline='') as csvfile:
#     csvwriter = csv.writer(csvfile)
#     # Write the header row
#     csvwriter.writerow(['Set', 'Image', 'Label', 'XMin', 'YMin', 'XMax', 'YMax'])

#     # Loop through each class directory within the set directory
#     for class_name in os.listdir(set_dir):
#         class_dir = os.path.join(set_dir, class_name)
#         images_dir = class_dir  # Images are directly under the class directory
#         labels_dir = 'OIDv/OID/label' # Annotations are in the 'Label' subdirectory

#         # Process annotations for each image in the selected set
#         for label_file in os.listdir(labels_dir):
#             image_file = label_file.replace('.txt', '.jpg')  # Adjust extension if needed
#             image_path = os.path.join(images_dir, image_file)

#             # Load the image to get its dimensions
#             with Image.open(image_path) as img:
#                 image_width, image_height = img.size

#             label_path = os.path.join(labels_dir, label_file)

#             # Read the annotation file, normalize, and write to CSV
#             with open(label_path, 'r') as file:
#                 lines = file.readlines()
#                 for line in lines:
#                     parts = line.strip().split(' ')
#                     label = parts[0]
#                     xmin, ymin, xmax, ymax = map(float, parts[1:])
#                     normalized_coords = normalize_bounding_box(xmin, ymin, xmax, ymax, image_width, image_height)

#                     # Write to CSV: set type, image file, label, and normalized coordinates
#                     csvwriter.writerow([selected_set_type, image_file, label, *normalized_coords])


import csv

list_column_2 = []
# Open the CSV file
with open("valid/_annotations.csv", "r") as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        list_column_2.append(row[1])

print(list_column_2)
# Skip the header row (optional)
# next(reader)

# Process each row
# for row in reader:


# import pandas as pd
import os

# # Load the provided CSV file to find out which images to keep
# csv_path = "valid/_annotations.csv"  # Replace with the path to the csv file
images_folder = "test"

# # Assuming the CSV file is correctly uploaded and named 'yourfile.csv'
# # Read the CSV file
# df = pd.read_csv(csv_path)
# # Get the list of image files from the CSV file
# images_to_keep = df.tolist()

# images_to_keep = images_to_keep[1]

# # Get the list of all image files in the folder
all_images = [img for img in os.listdir(images_folder) if img.endswith(".jpg")]

# # Find out which images to delete
images_to_delete = set(all_images) - set(list_column_2)

# # Delete the images that are not listed in the CSV file
for image in images_to_delete:
    os.remove(os.path.join(images_folder, image))

# # Return the number of deleted images for confirmation
len(images_to_delete)
