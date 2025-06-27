from tkinter import Tk, Label, Button, filedialog, font, Scrollbar, Frame, Canvas
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
import pandas as pd
from PIL import Image, ImageTk

red_list = pd.read_csv('red_list.csv')
bg_image = Image.open("background.jpg")
bg_image = bg_image.resize((1000, 700))

model = ResNet50(weights='imagenet')


def get_breed_status(file_path):
    img = image.load_img(file_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    decoded_preds = decode_predictions(preds, top=3)[0]
    breed_name = decoded_preds[0][1]
    endangered_status = red_list.loc[red_list['common name'] == breed_name]['red list status'].values
    if len(endangered_status) > 0:
        breed_result = "Predicted dog breed: " + breed_name
        status_result = "Endangered status: " + endangered_status[0]
        result = breed_result + "\n" + status_result
    else:
        result = "Predicted dog breed: " + breed_name + "\nEndangered status: Not Endangered"
    return result


def select_image():
    file_path = filedialog.askopenfilename()
    result = get_breed_status(file_path)
    label.config(text=result)
    img = Image.open(file_path)
    img.thumbnail((400, 400))
    img_tk = ImageTk.PhotoImage(img)
    img_label.config(image=img_tk)
    img_label.image = img_tk


def button_hover(event):
    button.config(bg='#20C1BD', fg='Aliceblue')


def button_reset(event):
    button.config(bg='#FDA5A4', fg='Aliceblue')


root = Tk()
root.title("Statistical Analysis Of Living Organisms")
root.geometry("620x400")
root.configure(bg='#FCE4D4')

my_font = font.Font(family='Montserrat', size=35, weight='bold')
smalie = font.Font(family='Khula', size=11)
smalie1 = font.Font(family='Khula', size=16)

canvas = Canvas(root, width=600, height=400, bg='#FCE4D4')
scrollbar = Scrollbar(root, orient="vertical", command=canvas.yview)
scrollable_frame = Frame(canvas, bg='#FCE4D4')

scrollable_frame.bind(
    "<Configure>",
    lambda e: canvas.configure(
        scrollregion=canvas.bbox("all")
    )
)

canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
canvas.configure(yscrollcommand=scrollbar.set)

canvas.pack(side="left", fill="both", expand=True)
scrollbar.pack(side="right", fill="y")

title_label = Label(scrollable_frame, text="The Dog Breed Detector", bg='#FCE4D4', fg='#DD7564', font=my_font)
title_label.pack()
title_label = Label(scrollable_frame, text="detect your favorite dogs with ease", bg='#FCE4D4', fg='#E6A57E', font=smalie)
title_label.pack()

label = Label(scrollable_frame, text="", bg='#FCE4D4', fg='#FF0000', font=('Montserrat', 20, 'bold'), justify='center')
label.pack()

img_label = Label(scrollable_frame, text="", bg='#FCE4D4', fg='#E6A57E', font=my_font)
img_label.pack()

button = Button(scrollable_frame, text="Select An Image", bg='#FC888C', fg='Aliceblue', font=smalie1, bd=0, activebackground='#FDA5A4', activeforeground='Aliceblue', command=select_image)
button.bind('<Enter>', button_hover)
button.bind('<Leave>', button_reset)
button.pack()

dog_label = Label(scrollable_frame, text="Dogs are a beloved pet all over the world, known for their loyalty and affection towards humans. They are a domesticated species of the Canidae family, originally descended from wolves. While they are commonly kept as house pets, dogs are also well-suited to living in a range of habitats, including forests, grasslands, and deserts. Their natural habitat varies depending on the breed, but most dogs prefer environments with moderate temperatures and access to water. \n\nDogs are found all over the world and are popular in most cultures. They are highly adaptable and can be found in almost any country or region, from the Arctic Circle to tropical islands. In terms of lifespan, the average dog can live up to 12 years, but this varies greatly depending on factors such as breed, size, and overall health. Some breeds can live up to 20 years or more, while others may only survive for a few years. Overall, dogs are a highly varied and interesting species with a long history of companionship with humans.", bg='#FCE4D4', fg='#000000', font=smalie, justify='left', pady=20, wraplength=500)
dog_label.pack()

root.mainloop()
