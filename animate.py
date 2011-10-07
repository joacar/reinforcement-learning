'''
Created on 5.10.2010.

@author: Tin Franovic
'''
from Tkinter import *
from PIL import Image, ImageTk

def do_animation(currentframe):
        def do_image():
                wrap.create_image(50,50,image=frame[currentframe])
        try:
                do_image()
        except IndexError:
                currentframe = 0
                do_image()
                wrap.update_idletasks()
        currentframe = currentframe + 1
        root.after(1000, do_animation, currentframe)

def draw(stateSequence):
    global wrap,frame,root
    root = Tk()
    root.title("WalkingRobot")
    frame=[]
    for i in stateSequence:
        fname="step"+str(i+1)+".png"
        img=ImageTk.PhotoImage(Image.open(fname))
        frame+=[img]

    wrap = Canvas(root, width=200, height=120)
    wrap.pack()
    root.after(10, do_animation, 0)
    root.mainloop()
