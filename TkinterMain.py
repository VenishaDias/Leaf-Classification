# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 14:04:01 2019

@author: Venisha Dias
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 13:43:14 2019

@author: Venisha Dias
"""

from tkinter import *
from tkinter import ttk
import csv
from tkinter import filedialog
from preprocessing import foreground,cropping
from PIL import Image, ImageTk
#from feature_extraction import shape,texture,color
from train_test import Test


def browsefunc(*args):
	filename = filedialog.askopenfilename()
	image_path = filename
	prediction = Test(image_path)
	with open('description.csv') as f:
		
		reader = csv.DictReader(f, delimiter=',')
		
		for row in reader:
			fn = row['no1']

			prediction1 = str(prediction)
			
			if(fn==prediction1):

				firstname = row['name']
				lastname = row['des']
				#print(type(lastname))

				leafname.set(firstname)
				leafsname.set("")
				descrip.set(lastname)
#				showimg(prediction) 
#			
#def showimg(prediction):
#	if(prediction == 1):
#			load = Image.open("Papaya-Leaves.jpg")
#			#fix = ((200,200))
#			#load = Image.resize(fix,0)
#			render = ImageTk.PhotoImage(load)
#	
#	        # labels can be text or images
#			img = ttk.Label(mainframe, image=render).grid(column = 5,row=1)
#			img.image = render
#			

        
			



	
    
root = Tk()
root.title('Botan.i')

mainframe = ttk.Frame(root, padding="3 3 500 500")
mainframe.grid(column=0, row=0, sticky=(N, W, E, S))
root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)
root.configure(background='green')


leafname = StringVar()
leafsname = StringVar()
descrip = StringVar()





ttk.Label(mainframe, textvariable=leafname ).grid(column=2, row=1, sticky=(W, E))
ttk.Label(mainframe, textvariable=leafsname).grid(column=2, row=2, sticky=(W, E))
ttk.Label(mainframe, textvariable=descrip).grid(column=2,columnspan=2,rowspan=3,row=3,sticky=(W, E))
#image.grid(row=0, column=2, columnspan=2, rowspan=2,sticky=W+E+N+S, padx=5, pady=5)
#imageEx = PhotoImage(file = 'image.gif')
#abel(leftFrame, image=imageEx).grid(row=0, column=0, padx=10, pady=2)
ttk.Button(mainframe, text="Identify", command=browsefunc).grid(column=2, row=60, sticky=W)


ttk.Label(mainframe, text="Plant" ).grid(column=1, row=1, sticky=W)
ttk.Label(mainframe, text="Scientific Name").grid(column=1, row=2, sticky=W)
ttk.Label(mainframe, text="Description").grid(column=1, row=3, sticky=W)

for child in mainframe.winfo_children(): child.grid_configure(padx=20, pady=20)

 
root.bind('<Return>', browsefunc)

root.mainloop()