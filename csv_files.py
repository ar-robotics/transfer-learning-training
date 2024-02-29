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

# Open the CSV file
with open("only_people.csv", "r") as csvfile:
  reader = csv.reader(csvfile)

  # Skip the header row (optional)
  #next(reader)

  # Process each row
  for row in reader:
      
      # Extract the coordinates
      #x1, y1, x2, y2 = row[3], row[4], row[7], row[8]
      #print("before",x1, y1, x2, y2)
      # Round down coordinates exceeding 1
    #   x1 = min(float(x1), 1)
    #   y1 = min(float(y1), 1)
    #   x2 = min(float(x2), 1)
    #   y2 = min(float(y2), 1)
      #print("after",x1, y1, x2, y2)
      # Reconstruct the row with modified coordinates
      updated_row = [row[0], row[1], row[2],row[3],row[4], "","", row[7], row[8], ""]

      # Print or write the updated row as needed
      print(updated_row)  # Example: print the updated row
      with open("people_updated_labels.csv", "a", newline="") as new_file:
           writer = csv.writer(new_file)
           writer.writerow(updated_row)
      #Alternatively, write the updated row to a new CSV file

        

