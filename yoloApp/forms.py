from django import forms  
class ImageForm(forms.Form):  
    file  = forms.ImageField(label="Enter the vehicle image") 