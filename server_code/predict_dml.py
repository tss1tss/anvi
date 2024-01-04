# import anvil.server

# # *********************** DML ************************
# # *******************
# ##matplotlib inline
# from pytorch_metric_learning.utils.inference import MatchFinder, InferenceModel
# from pytorch_metric_learning.distances import CosineSimilarity
# from pytorch_metric_learning.utils import common_functions as c_f
# from torchvision import datasets, transforms
# import torchvision
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.nn.init as init
# from torch.utils.data import Dataset, DataLoader
# from torchvision.transforms import ToPILImage, PILToTensor
# import torchvision.models as models

# import matplotlib.pyplot as plt
# import numpy as np
# import glob
# from PIL import Image
# import csv
# from pandas.core.common import flatten
# import cv2
# import random


# train_data_path = '/home/tss2tss/Drive_TB/ai/DML/DML_pH_dataset/DML_train' 
# train_image_paths = []
# classes = []

# for data_path in glob.glob(train_data_path + '/*'):
#     classes.append(data_path.split('/')[-1])
#     train_image_paths.append(glob.glob(data_path + '/*.jpg'))
    
# train_image_paths = list(flatten(train_image_paths))
# random.shuffle(train_image_paths)

# print('train_image_path example: ', train_image_paths[1])
# print('class example: ', classes[1])


# train_image_paths, valid_image_paths = train_image_paths[:int(0.8*len(train_image_paths))], train_image_paths[int(0.8*len(train_image_paths)):] 


# train_transform = transforms.Compose([transforms.Resize(32),
#                                     transforms.RandomResizedCrop(scale=(0.16, 1), ratio=(0.75, 1.33), size=64),
#                                     transforms.RandomHorizontalFlip(0.5),
#                                     transforms.ToTensor(),
#                                     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

# test_transform = transforms.Compose([transforms.Resize(32),
#                                     transforms.ToTensor(),])

# idx_to_class = {i:j for i, j in enumerate(classes)}
# class_to_idx = {value:key for key,value in idx_to_class.items()}

# class TyDataset(Dataset):
#     def __init__(self, image_paths, transform=False):
#         self.image_paths = image_paths
#         self.image_dim = (32, 32)
#         self.transform = transform
        
#     def __len__(self):
#         return len(self.image_paths)

#     def __getitem__(self, idx):              
#         image_filepath = self.image_paths[idx]
#         image = cv2.imread(image_filepath)
#         image = cv2.resize(image, self.image_dim)
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         image = torch.from_numpy(image)
#         image = image.permute(2,0,1)
#         image = image.float()
#         label = image_filepath.split('/')[-2]
#         label = class_to_idx[label]
        
#         return image, label

# train_dataset = TyDataset(train_image_paths,train_transform)
# targets_train = []
# for _, labels in  train_dataset:
#     targets_train.append(labels)
# labels_to_indices_train = c_f.get_labels_to_indices(targets_train)
# class MLP(nn.Module):

#     def __init__(self, layer_sizes, final_relu=False):
#         super().__init__()
#         layer_list = []
#         layer_sizes = [int(x) for x in layer_sizes]
#         num_layers = len(layer_sizes) - 1
#         final_relu_layer = num_layers if final_relu else num_layers - 1
#         for i in range(len(layer_sizes) - 1):
#             input_size = layer_sizes[i]
#             curr_size = layer_sizes[i + 1]
#             if i < final_relu_layer:
#                 layer_list.append(nn.ReLU(inplace=False))
#             layer_list.append(nn.Linear(input_size, curr_size))
#         self.net = nn.Sequential(*layer_list)
#         self.last_linear = self.net[-1]

#     def forward(self, x):
#         return self.net(x)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# trunk = models.resnet152()
# trunk_output_size = trunk.fc.in_features
# trunk.fc = nn.Identity()
# trunk.to(device)
# trunk = torch.nn.DataParallel(trunk)
# embedder = torch.nn.DataParallel(MLP([trunk_output_size, 2048]))
# embedder.to(device)

# trunk.module.load_state_dict(torch.load('/home/tss2tss/Drive_TB/ai/DML/Res152_800ep_x2048/saved_models/trunk_best739.pth'))
# embedder.module.load_state_dict(torch.load('/home/tss2tss/Drive_TB/ai/DML/Res152_800ep_x2048/saved_models/embedder_best739.pth'))


# print("done model loading")
# trunk.eval()
# embedder.eval()
# match_finder = MatchFinder(distance=CosineSimilarity(), threshold=0.5)
# inference_model = InferenceModel(trunk, embedder, match_finder)
# inference_model.train_knn(train_dataset)



# class_indices = [labels_to_indices_train[i] for i in range(len(idx_to_class))]
# class SingleImageDataset(Dataset):
#     def __init__(self, image_path):
#         self.image_path = image_path
#         self.image_dim = (32, 32)

#     def __len__(self):
#         return 1

#     def __getitem__(self, idx):              
#         image = cv2.imread(self.image_path)
#         image = cv2.resize(image, self.image_dim)
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         image = torch.from_numpy(image)
#         image = image.permute(2,0,1)
#         image = image.float()
#         return image
# def imshow(img, label, label1, figsize=(2, 1)):
#     npimg = img.cpu().numpy()
#     plt.figure(figsize = figsize)
#     plt.imshow(np.transpose(npimg, (1, 2, 0)).astype(np.uint8))
#     plt.title(label)
#     plt.xlabel(label1)
#     plt.tick_params(
#     axis='both',          # changes apply to the x-axis
#     which='both',      # both major and minor ticks are affected
#     labelleft=False,      # ticks along the bottom edge are off
#     labelbottom=False) # labels along the bottom edge are off 
# def imshow1(img, label, label1, figsize=(14, 7)):
#     npimg = img.numpy()
#     plt.figure(figsize = figsize)
#     plt.imshow(np.transpose(npimg, (1, 2, 0)).astype(np.uint8))
#     plt.title(label)
#     plt.xlabel(label1)
#     plt.tick_params(
#     axis='both',          # changes apply to the x-axis
#     which='both',      # both major and minor ticks are affected
#     labelleft=False,      # ticks along the bottom edge are off
#     labelbottom=False) # labels along the bottom edge are off 

# def imshow2(img, label, label1, figsize=(2, 1)):
#     npimg = img.cpu().numpy()
#     plt.figure(figsize = figsize)
#     plt.imshow(np.transpose(npimg, (1, 2, 0)).astype(np.uint8))
#     plt.title(label)
#     plt.xlabel(label1)
#     plt.tick_params(
#     axis='both',          # changes apply to the x-axis
#     which='both',      # both major and minor ticks are affected
#     labelleft=False,      # ticks along the bottom edge are off
#     labelbottom=False) # labels along the bottom edge are off 


# kkk = 20  # Number of nearest images
# actual_class = 'pH4.0'
# # image_path = "/home/tss2tss/Drive_TB/ai/DML/DML_pH_dataset/DML_test_2/pH4.0/48.jpg"
# # image_path = image_path

# @anvil.server.callable
# def predict_dml(img_path):
#     image_path = img_path
#     single_image_dataset = SingleImageDataset(image_path)
#     img_tensor = single_image_dataset[0]  # Fetch the image tensor
#     img = img_tensor.unsqueeze(0).to(device)  # Add batch dimension and move to device
#     imshow(torchvision.utils.make_grid(img), label="Query", label1=actual_class) ###change class name
#     distances, indices = inference_model.get_nearest_neighbors(img, k=kkk)
#     # print("indices : ", indices)
#     # print("distances", distances)
#     sorted_pairs = sorted(zip(distances, indices), key=lambda x: x[0], reverse=True)
#     sorted_distances, sorted_indices = zip(*sorted_pairs)

#     nearest_imgs = [train_dataset[i][0] for i in indices.cpu()[0]]
#     percentages = []
#     for class_idx in class_indices:
#         result = np.in1d(indices.cpu(), class_idx)
#         count = list(result).count(True)
#         percentage = count / kkk
#         percentages.append(percentage)
#     max_percentage = max(percentages)
#     max_percentage_idx = percentages.index(max(percentages))
#     predicted_classes = [idx_to_class[i] for i, p in enumerate(percentages) if p == max_percentage]

#     is_correct = 1 if actual_class in predicted_classes else 0
#     predicted_class = actual_class if is_correct == 1 else predicted_classes[0]
#     data = [actual_class, predicted_class, is_correct, max_percentage]
        
#     with open('/home/tss2tss/Drive_TB/ai/DML/pH8.0.csv', 'a', encoding='UTF8') as f:  # Change save file name
#         writer = csv.writer(f)
#         header = ['Actual', 'Predict', 'T(1)/F(0)', 'Percent']
#         writer.writerow(header)
#         writer.writerow(data)
#     print("end")


#     out = torchvision.utils.make_grid(nearest_imgs, nrow=10)
#     result = np.in1d(indices.cpu(), labels_to_indices_train[max_percentage_idx])  #compare with the train set
#     dis = [(result[x]) for x in range(len(result))]
#     # print("result : ", result)
#     # print("dis : ", dis)
#     predicted_class = actual_class if is_correct == 1 else predicted_classes[0]
#     imshow1(out, label="Nearest", label1 = "Results = %s"  % dis)
#     imshow2(torchvision.utils.make_grid(img), label="Predict", label1 = predicted_class + " (" + "%0.2f" % max_percentage + ")")
#     plt.show()


# # anvil.server.connect("server_NNQ4KD2P22BKV7R4ZQ6G4OBQ-N4W6O2NSC63FJKR2")

# # # # # # # # Keep the server running
# # anvil.server.wait_forever()