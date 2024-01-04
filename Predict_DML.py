import anvil.server
from io import BytesIO
import anvil.media
anvil.server.connect("server_WLBRH5HOXJMZ2UTBHXK6VDMG-N4W6O2NSC63FJKR2", url="ws://localhost:3030/_/uplink")

# *********************** DML ************************
# *******************
##matplotlib inline
from pytorch_metric_learning.utils.inference import MatchFinder, InferenceModel
from pytorch_metric_learning.distances import CosineSimilarity
from pytorch_metric_learning.utils import common_functions as c_f
from torchvision import datasets, transforms
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToPILImage, PILToTensor
import torchvision.models as models

import matplotlib.pyplot as plt
import numpy as np
import glob
from PIL import Image
import csv
from pandas.core.common import flatten
import cv2
import random
# anvil.server.connect("server_5LB5N23TAH6KRBFK3FUPUEZI-N4W6O2NSC63FJKR2", url="ws://localhost:3030/_/uplink")
model = None
embedder = None
inference_model = None
train_dataset = None
class_indices = None

@anvil.server.callable
def load_model(img_size, backbone, train_data_path, model_path, embedder_path, embedder_size):
    train_data_path = train_data_path
    train_image_paths = []
    global classes
    classes = []

    for data_path in glob.glob(train_data_path + '/*'):
        classes.append(data_path.split('/')[-1])
        train_image_paths.append(glob.glob(data_path + '/*.jpg'))

    train_image_paths = list(flatten(train_image_paths))
    random.shuffle(train_image_paths)

    print('train_image_path example: ', train_image_paths[1])
    print('class example: ', classes[1])

    # train_image_paths, valid_image_paths = train_image_paths[:int(0.8 * len(train_image_paths))], train_image_paths[int(0.8 * len(train_image_paths)):]
    train_image_paths, _ = train_image_paths[:int(0.8 * len(train_image_paths))], train_image_paths[int(0.8 * len(train_image_paths)):]

    train_transform = transforms.Compose([
        transforms.Resize(32),
        transforms.RandomResizedCrop(scale=(0.16, 1), ratio=(0.75, 1.33), size=64),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
    ])
    global idx_to_class
    idx_to_class = {i: j for i, j in enumerate(classes)}
    class_to_idx = {value: key for key, value in idx_to_class.items()}

    class TyDataset(Dataset):
        def __init__(self, image_paths, transform=False):
            self.image_paths = image_paths
            self.image_dim = (img_size, img_size)
            self.transform = transform

        def __len__(self):
            return len(self.image_paths)

        def __getitem__(self, idx):
            image_filepath = self.image_paths[idx]
            image = cv2.imread(image_filepath)
            image = cv2.resize(image, self.image_dim)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = torch.from_numpy(image)
            image = image.permute(2, 0, 1)
            image = image.float()
            label = image_filepath.split('/')[-2]
            label = class_to_idx[label]

            return image, label
    global train_dataset
    train_dataset = TyDataset(train_image_paths, train_transform)
    targets_train = []
    for _, labels in train_dataset:
        targets_train.append(labels)
    global labels_to_indices_train
    labels_to_indices_train = c_f.get_labels_to_indices(targets_train)

    class MLP(nn.Module):
        def __init__(self, layer_sizes, final_relu=False):
            super(MLP, self).__init__()
            layer_list = []
            layer_sizes = [int(x) for x in layer_sizes]
            num_layers = len(layer_sizes) - 1
            final_relu_layer = num_layers if final_relu else num_layers - 1
            for i in range(len(layer_sizes) - 1):
                input_size = layer_sizes[i]
                curr_size = layer_sizes[i + 1]
                if i < final_relu_layer:
                    layer_list.append(nn.ReLU(inplace=False))
                layer_list.append(nn.Linear(input_size, curr_size))
            self.net = nn.Sequential(*layer_list)
            self.last_linear = self.net[-1]

        def forward(self, x):
            return self.net(x)

    # ********* load model **********

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def get_model(backbone):
        models_dict = {
            "resnet50": models.resnet50,
            "resnet101": models.resnet101,
            "resnet152": models.resnet152
        }

        model_func = models_dict.get(backbone)
        if model_func:
            return model_func()
        else:
            raise ValueError("Invalid backbone specified")
    # model = get_model(backbone)
   
    # model_output_size = model.fc.in_features
    # model.fc = nn.Identity()
    # model.to(device)
    # model = torch.nn.DataParallel(model)
    # embedder = torch.nn.DataParallel(MLP([model_output_size, embedder_size]))
    # embedder.to(device)
    # model.module.load_state_dict(torch.load(model_path))
    # embedder.module.load_state_dict(torch.load(embedder_path))


    model = get_model(backbone)
    model_output_size = model.fc.in_features
    model.fc = nn.Identity()
    model.to(device)
    model.load_state_dict(torch.load(model_path))
    embedder = MLP([model_output_size, embedder_size])
    embedder.to(device)
    embedder.load_state_dict(torch.load(embedder_path))

    print("done model loading")
    global inference_model
    model.eval()
    embedder.eval()
    match_finder = MatchFinder(distance=CosineSimilarity(), threshold=0.5)
    inference_model = InferenceModel(model, embedder, match_finder)
    inference_model.train_knn(train_dataset)
    global class_indices
    class_indices = [labels_to_indices_train[i] for i in range(len(idx_to_class))]
    model = None
    print("Load model and embedder success")
    have_model = True
    return have_model, class_indices
@anvil.server.callable
def release_model():
    
    global model, embedder, inference_model, train_dataset, class_indices

    try:
        # Release models
        if model is not None:
            del model
            model = None
        if embedder is not None:
            del embedder
            embedder = None
        if inference_model is not None:
            del inference_model
            inference_model = None

        torch.cuda.empty_cache()  # Clear GPU cache if models were on GPU

        # Release other resources
        train_dataset = None
        class_indices = None

        print("Model and resources released successfully")

    except Exception as e:
        print(f"Error releasing model: {e}")


        
@anvil.server.callable
def predict_dml(img_path, k):
    
    class SingleImageDataset(Dataset):
        def __init__(self, image_path):
            self.image_path = image_path
            self.image_dim = (32, 32)

        def __len__(self):
            return 1

        def __getitem__(self, idx):              
            image = cv2.imread(self.image_path)
            image = cv2.resize(image, self.image_dim)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = torch.from_numpy(image)
            image = image.permute(2,0,1)
            image = image.float()
            return image
    
    kkk = int(k)
    image_path = img_path
    single_image_dataset = SingleImageDataset(image_path)
    img_tensor = single_image_dataset[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img = img_tensor.unsqueeze(0).to(device)  
    distances, indices = inference_model.get_nearest_neighbors(img, k=kkk)
    # image_number = indices.cpu()[0][0].item()
    reslut_class_name = []
    for image_number in indices.cpu()[0]:
        image_number = image_number.item()
        for i, indices_1 in enumerate(class_indices):
            if image_number in indices_1:
                reslut_class_name.append(idx_to_class[i])
    
    # print("*********** indices_class_name : ", reslut_class_name)
                
    # sorted_pairs = sorted(zip(distances, indices), key=lambda x: x[0], reverse=True)
    # sorted_distances, sorted_indices = zip(*sorted_pairs)
    nearest_imgs = [train_dataset[i][0] for i in indices.cpu()[0]]
    percentages = []
    for class_idx in class_indices:
        result = np.in1d(indices.cpu(), class_idx)
        count = list(result).count(True)
        percentage = count / kkk
        percentages.append(percentage)
    max_percentage = max(percentages)
    max_percentage_idx = percentages.index(max(percentages))
    predicted_classes = [idx_to_class[i] for i, p in enumerate(percentages) if p == max_percentage]
    reslut_predicted_classes = f"{predicted_classes} ({max_percentage:.2f})"

    out_1 = torchvision.utils.make_grid(nearest_imgs, nrow=int(int(kkk)/2) + 1)
    result = np.in1d(indices.cpu(), labels_to_indices_train[max_percentage_idx])  #compare with the train set
    # dis = [(result[x]) for x in range(len(result))]

    img_bytes_1 = BytesIO()
    # img_bytes_2 = BytesIO()

    # Create the first image
    out_1 = torchvision.utils.make_grid(nearest_imgs)
    # plt.imshow(np.transpose(out_1.cpu().numpy(), (1, 2, 0)).astype(np.uint8))
    # label_1 = "Nearest"
    # label1_1 = "Results = %s" % reslut_class_name
    # plt.title(label_1)
    # plt.xlabel(label1_1)
    # plt.tick_params(
    #     axis='both',
    #     which='both',
    #     labelleft=False,
    #     labelbottom=False)
    # plt.savefig(img_bytes_1, format='png', bbox_inches='tight', pad_inches=0.0)
    # plt.close()
    image = np.transpose(out_1.cpu().numpy(), (1, 2, 0)).astype(np.uint8)
    pil_img = Image.fromarray(image)
    pil_img.save(img_bytes_1, format='png')

    return anvil.BlobMedia('image/png', img_bytes_1.getvalue()), reslut_class_name, reslut_predicted_classes

anvil.server.wait_forever()