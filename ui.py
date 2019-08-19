import shakespeareGen
from tkinter import *

def updateText():
    numChars = numCharsSlider.get()
    
    textOutput.config(state=NORMAL)
    
    textOutput.delete(0.0, END)
    textOutput.insert(END, shakespeareGen.generate(numChars))
    
    textOutput.config(state=DISABLED)

window = Tk()
window.title("Makespeare")
window.minsize(500, 200)

########## Widget creation

titleFrame = Frame(window)

titleLabel = Label(titleFrame, text="Makespeare 1.0", font="arial 14 bold")
creditLabel = Label(titleFrame, text="Shakespearean text generator by Borna Sadeghi", font="arial 10")

inputFrame = Frame(window)

numCharsLabel = Label(inputFrame, text="Number of characters to generate:")
numCharsSlider = Scale(inputFrame, from_=100, to=5000, orient=HORIZONTAL, length=400)
numCharsSlider.set(1000)

generateButton = Button(window, text="Generate", command=updateText)
textOutput = Text(window, state=DISABLED)

########## Widget placement

titleLabel.grid(row=0, column=0, sticky=W)
creditLabel.grid(row=0, column=1, sticky=E)

titleFrame.pack(side=TOP)

numCharsLabel.grid(row=0, column=0)
numCharsSlider.grid(row=0, column=1)

inputFrame.pack(side=TOP, pady=20)

generateButton.pack(side=TOP)
textOutput.pack()


window.mainloop()