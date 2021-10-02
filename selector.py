"""
Created on Jul  2 09:50:44 2021

@author: Ámbar Pérez García
"""
import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
#from comparador import path
import json

path = "data\HICO_1"
WIDTH, HEIGHT = 1000, 1000
topx, topy, botx, boty = 0, 0, 0, 0
rect_id = None
vector = np.array([[0,0,0,0]]) 
dic = {}

def get_mouse_posn(event):
    global topy, topx

    topx, topy = event.x, event.y

def update_sel_rect(event):
    global rect_id
    global topy, topx, botx, boty

    botx, boty = event.x, event.y
    canvas.coords(rect_id, topx, topy, botx, boty)  # Update selection rect.

def zoom(event):
    if (event.delta > 0):
        canvas.scale("all", event.x, event.y, 1.1, 1.1)
    elif (event.delta < 0):
        canvas.scale("all", event.x, event.y, 0.9, 0.9)
    canvas.configure(scrollregion = canvas.bbox("all"))

def store_coordinates(event=None):
    global vector
    if vector[0][1] == 0: vector[0] = np.array([topy,boty,topx,botx])
    else: vector = np.append(vector, [[topy,boty,topx,botx]], axis=0)
    print(f"{topy=}, {boty=}, {topx=}, {botx=}")

def convert_coordinates(vector, dim): # Coordinates are converted to dimension 1
    nvector = np.array([])
    for i in vector:
        for j in range(i[0], i[1]+1):
            nvector = np.append(nvector, np.arange(dim*j + i[2], dim*j + i[3] + 1))
    return nvector

def json_store(path, pixels):
    with open(path + ".json", "w") as write_file:
        json.dump(pixels, write_file)


for i in ["Water", "Oil"]: # Select different classes
    # Create window to work
    window = tk.Tk()
    window.title("Select Area:" + i)
    window.geometry('%sx%s' % (WIDTH, HEIGHT))
    window.configure(background='grey')

    img = ImageTk.PhotoImage(Image.open(path + ".jpeg"))
    canvas = tk.Canvas(window, width=img.width(), height=img.height(), borderwidth=0, highlightthickness=0)
    canvas.create_image(0, 0, image=img, anchor=tk.NW)
    canvas.pack(expand=True)

    # Create selection rectangle 
    rect_id = canvas.create_rectangle(topx, topy, topx, topy, dash=(2,2), fill='', outline='white')

    canvas.bind('<Button-1>', get_mouse_posn)
    canvas.bind('<B1-Motion>', update_sel_rect)
    canvas.bind('<Button-3>', store_coordinates)
    canvas.bind('<MouseWheel>', zoom)
    canvas.pack()

    window.mainloop()

    dic[i] = convert_coordinates(vector, img.width()).tolist()
    vector = np.array([[0,0,0,0]]) 


json_store(path, dic)

    