from ._anvil_designer import homeTemplate
from anvil import *
import anvil.server

class home(homeTemplate):
  def __init__(self, **properties):
    # Set Form properties and Data Bindings.
    self.init_components(**properties)

  def button_logout_click(self, **event_args):
    """This method is called when the button is clicked"""
    anvil.open_form('login')
    pass

  def bt_dml_and_retrieval_click(self, **event_args):
    """This method is called when the button is clicked"""
    anvil.open_form('home.dml')
    anvil.server.call('say_hello', 'tae')
    pass

 
    
