from ._anvil_designer import loginTemplate
from anvil import *
import anvil.server
from ..home import home

class login(loginTemplate):
  def __init__(self, **properties):
    # Set Form properties and Data Bindings.
    self.init_components(**properties)

  def input_password_hide(self, **event_args):
    """This method is called when the text area is removed from the screen"""
    pass

  def login_click(self, **event_args):
    """This method is called when the button is clicked"""
    username = self.input_username.text
    password = self.input_password.text
    print(username, password)
    if(username == 'admin' and password == 'cira2023'):
      anvil.open_form('home')
      print('login success')
    else:
      print('User Name or Password Is Incorrect')
    pass