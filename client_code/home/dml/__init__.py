from ._anvil_designer import dmlTemplate
from anvil import *
import anvil.server
    
class dml(dmlTemplate):
  def __init__(self, **properties):
    
    # Set Form properties and Data Bindings.
    self.init_components(**properties)
    # self.label_show_upload.border = '1px dashed black'
    self.label_image_name.border = '1px dashed black'
    self.file_loader_img.accept = ['.jpg', '.jpeg', '.png', '.gif']
    self.drop_down_models.items = ['Custom model', 'Trypanosome', 'pH_reader']
    self.k_number.text = 10
    ## for custom models 
    self.label_custom_model.visible = True
    self.label_embedder.visible = True
    self.file_loader_embedder.visible = True
    self.file_loader_model.visible = True
    self.label_embedder_name.visible =True
    self.label_model_name.visible = True
    self.have_model = False
    
    self.uploaded = False
    self.folder = anvil.server.call('create_first_folder')

  def dml_click(self, **event_args):
    """This method is called when the link is clicked"""
    anvil.server.call('say_hello', 'images')
    pass

  def button_2_click(self, **event_args):
    """This method is called when the button is clicked"""
    anvil.open_form('login')
    pass

  def bt_dml_and_retrieval_click(self, **event_args):
    """This method is called when the button is clicked"""
    anvil.open_form('home.dml')
    pass

  def bt_home_click(self, **event_args):
    """This method is called when the button is clicked"""
    anvil.open_form('home')
    pass

  def file_loader_img_change(self, **event_args):
      """This method is called when a new file is loaded into this FileLoader"""
      uploaded_files = self.file_loader_img.files
      if uploaded_files:
          # Check if all files are images
        all_images = all(file.content_type.startswith('image/') for file in uploaded_files)
        
        if all_images:
          self.curr_img = 1
          self.img_size, self.image_files, image_folder = anvil.server.call('save_images_to_server', uploaded_files, self.folder)
          print('img size : ', self.img_size)
          anvil.Notification("Images are being uploaded...").show()
          # img size
          self.img_all.text = int(self.img_size)
          self.img_now.text = self.curr_img
          image_media = anvil.server.call('get_image', self.image_files[self.curr_img - 1])
          
          if image_media:
            self.image_query.source = image_media
            # self.label_show_upload.text = "To Prediction."
            self.label_image_name.text = self.image_files[self.curr_img - 1]
            self.uploaded = 1
            k = int(self.k_number.text)
            if k is not None and k != '' and self.have_model == True:
              # img_blob, result_img = anvil.server.call('predict_dml', self.image_files[self.curr_img - 1], k)
              result_img, reslut_class_name, reslut_predicted_classes = anvil.server.call('predict_dml', self.image_files[self.curr_img - 1], k)
              self.label_result_write.text = reslut_class_name
              self.image_predict.source = result_img
            
          else:
            anvil.Notification("Error can't load image").show()
            # self.label_show_upload.text = "Error from show image"
        else:
          anvil.Notification("Please upload only image files.").show()
          # self.label_show_upload.text = "Please upload only image files."

  def bt_next_img_click(self, **event_args):
    print("bt_next_img_click")
    """This method is called when the button is clicked"""
    if(self.uploaded):
      if(self.curr_img == self.img_size):
        self.curr_img = 0
      if(self.curr_img < self.img_size > 1):
        self.curr_img = self.curr_img + 1
        self.img_now.text = self.curr_img
        image_media = anvil.server.call('get_image', self.image_files[self.curr_img - 1])
        if image_media:
          self.image_query.source = image_media
          self.label_image_name.text = self.image_files[self.curr_img - 1]
          k = int(self.k_number.text)
          if k is not None and k != '':
            if(self.have_model):
              # img_blob, result_img = anvil.server.call('predict_dml', self.image_files[self.curr_img - 1], k)
              result_img, reslut_class_name, reslut_predicted_classes = anvil.server.call('predict_dml', self.image_files[self.curr_img - 1], k)
              self.label_result_write.text = reslut_class_name
              self.image_predict.source = result_img
              # self.image_k.source = img_blob
          else:
            anvil.Notification("TextBox value is None, empty, or undefined.").show()
        else:
          anvil.Notification("n Error can't load image").show()
    else:
      anvil.Notification("Please upload image files.").show()

  def bt_previous_img_click(self, **event_args):
    print("bt_next_img_click")

    if(self.curr_img == 1):
        self.curr_img = self.img_size + 1
    if(self.uploaded):
      if(self.img_size != 1 and self.curr_img > 1):
        self.curr_img = self.curr_img - 1
        self.img_now.text = self.curr_img
        image_media = anvil.server.call('get_image', self.image_files[self.curr_img - 1])
        if image_media:
          self.image_query.source = image_media
          self.label_image_name.text = self.image_files[self.curr_img - 1]
          k = int(self.k_number.text)
          if k is not None and k != '':
            if(self.have_model):
              # img_blob, result_img = anvil.server.call('predict_dml', self.image_files[self.curr_img - 1], k)
              result_img, reslut_class_name, reslut_predicted_classes = anvil.server.call('predict_dml', self.image_files[self.curr_img - 1], k)
              self.label_result_write.text = reslut_class_name
              self.image_predict.source = result_img
              # self.image_k.source = img_blob
          else:
            anvil.Notification("TextBox value is None, empty, or undefined.").show()
        else:
          anvil.Notification("n Error can't load image").show()
    else:
      anvil.Notification("Please upload image files.").show()

  def drop_down_models_change(self, **event_args):
    """This method is called when an item is selected"""
    if(self.drop_down_models.selected_value == 'Custom model'):
      self.label_custom_model.visible = True
      self.label_embedder.visible = True
      self.file_loader_embedder.visible = True
      self.file_loader_model.visible = True
      self.label_embedder_name.visible =True
      self.label_model_name.visible = True
      self.label_upload_trained_img.visible = True
      self.file_loader_trained_img.visible = True
      self.label_train_img.visible = True
      self.class_indices = anvil.server.call('release_model')
      
    if(self.drop_down_models.selected_value == 'pH_reader'):
      self.label_custom_model.visible = False
      self.label_embedder.visible = False
      self.file_loader_embedder.visible = False
      self.file_loader_model.visible = False
      self.label_embedder_name.visible =False
      self.label_model_name.visible = False
      self.label_upload_trained_img.visible = False
      self.file_loader_trained_img.visible = False
      self.label_train_img.visible = False
      self.label_backbone.visible = False
      self.drop_down_backbone.visible = False
      self.class_indices = anvil.server.call('release_model')
      train_data_path = '/home/tss2tss/Drive_TB/ai/DML/DML_pH_dataset/DML_train'
      model_path = '/home/tss2tss/Drive_TB/ai/DML/Res152_800ep_x2048/saved_models/trunk_best739.pth'
      embedder_path = '/home/tss2tss/Drive_TB/ai/DML/Res152_800ep_x2048/saved_models/embedder_best739.pth'
      self.have_model, self.class_indices = anvil.server.call('load_model', 32, 'resnet152', train_data_path, model_path, embedder_path, 2048)
      # self.class_indices = anvil.server.call('load_model', img_size, backbone, train_data_path, model_path, embedder_path, embedder_size) 
    
    if(self.drop_down_models.selected_value == 'Trypanosome'):
      self.label_custom_model.visible = False
      self.label_embedder.visible = False
      self.file_loader_embedder.visible = False
      self.file_loader_model.visible = False
      self.label_embedder_name.visible =False
      self.label_model_name.visible = False
      self.label_upload_trained_img.visible = False
      self.file_loader_trained_img.visible = False
      self.label_train_img.visible = False
      self.label_backbone.visible = False
      self.drop_down_backbone.visible = False 

      self.class_indices = anvil.server.call('release_model')
      train_data_path = '/home/tss2tss/Drive_TB/ai/DML/models/ResNet50_TripletMargin/Train'
      model_path = '/home/tss2tss/Drive_TB/ai/DML/models/ResNet50_TripletMargin/Metric_Learning/saved_models/trunk_best199.pth'
      embedder_path = '/home/tss2tss/Drive_TB/ai/DML/models/ResNet50_TripletMargin/Metric_Learning/saved_models/embedder_best199.pth'
      self.have_model, self.class_indices = anvil.server.call('load_model', 32, 'resnet50', train_data_path, model_path, embedder_path, 64)
      
    pass
# anvil.server.connect("[uplink-key goes here]", url="ws://your-runtime-server:3030/_/uplink")
