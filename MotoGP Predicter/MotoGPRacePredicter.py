# Import dependencies
from tkinter import messagebox, filedialog, filedialog, Tk, Label, Button
import os
from race_predicter import trainModel,predictModel

# Window config
window = Tk()
window.title("Moto GP Race Predicter")
window.geometry('325x125')

# App label
lbl=Label(window, text="Place your bets!",font=("Arial Bold", 30)) 
lbl.grid(column=0, row=0)

# Train model
def train():
    file=filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = (("csv files","*.csv"),("all files","*.*")))
    messagebox.showinfo("Model Trained!", "Check terminal window for accuracy scores!")
    trainModel(file)
    
# Training button
btn = Button(window,text="Train Model",command=train)
btn.grid(column=0,row=1)

# Test new data
def predict():
    file=filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = (("csv files","*.csv"),("all files","*.*")))
    messagebox.showinfo("Prediction Complete", "Check terminal window for predicted winners!")
    predictModel(file)

btn = Button(window,text="Predict Race",command=predict)
btn.grid(column=0,row=2)

# Setting mainloop, (keeps the window running until user input)
window.mainloop()