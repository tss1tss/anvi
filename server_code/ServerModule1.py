import anvil.server
import os
import time
from datetime import datetime
from io import BytesIO
import matplotlib.pyplot as plt

# This is a server module. It runs on the Anvil server,
# rather than in the user's browser.
#
# To allow anvil.server.call() to call functions here, we mark
# them with @anvil.server.callable.
# Here is an example - you can replace it with your own:
#

@anvil.server.callable
def say_hello(name):
  print("Hello, " + name + "!")
  return 42

# @anvil.server.callable
# def save_images_to_server(files):
#     # Create a timestamped folder
#     timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
#     folder_path = f'{timestamp}'
#     os.makedirs(folder_path, exist_ok=True)
#     # Save each file in the timestamped folder
#     for file in files:
#         file_path = os.path.join(folder_path, file.name)
#         with open(file_path, 'wb') as f:
#             f.write(file.get_bytes())
    
#     return f"All images have been saved in folder: {folder_path}", folder_path


# Global variable to keep track of the images and current index

@anvil.server.callable
def  create_first_folder():
    folder = datetime.now().strftime('%Y%m%d%H%M%S')
    folder = f'folder_{folder}'
    os.makedirs(folder, exist_ok=True)
    return folder
image_folder = None
image_files = []
current_index = 0

@anvil.server.callable
def save_images_to_server(uploaded_files, folder):
    global image_folder, image_files, current_index
    
    # Create a timestamped folder
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    image_folder = folder + '/' + f'{timestamp}'
    os.makedirs( image_folder, exist_ok=True)
    
    # Save each file in the timestamped folder and keep track of the files
    image_files = []
    for file in uploaded_files:
        file_path = os.path.join(image_folder, file.name)
        with open(file_path, 'wb') as f:
            f.write(file.get_bytes())
        image_files.append(file_path)
    
    return len(image_files),  image_files,  image_folder# Return the number of images saved

@anvil.server.callable
def get_image(image_files):
  with open(image_files, 'rb') as f:
      return anvil.BlobMedia('image/*', f.read(), name=os.path.basename(image_files))

@anvil.server.callable
def plt_to_anvil(plt_img):
    bytes_io = BytesIO()
    plt_img.savefig(bytes_io, format='png')  # This does not save to disk, it saves to an in-memory stream
    bytes_io.seek(0)  # Reset the pointer to the start of the stream
    print('plttttt')
    media_obj = anvil.media.from_file(bytes_io, 'image/png')
    return media_obj

