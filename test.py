import os
from tkinter import *
from MNISTTester import MNISTTester
from paint import Paint

####################
# directory settings
script_dir = os.path.dirname(os.path.abspath(__file__))

data_path = script_dir + '/mnist/data/'
model_path = script_dir + '/models/mnist-cnn'

#####################################
# prediction test with MNIST test set
mnist = MNISTTester(
            model_path=model_path,
            data_path=data_path)
def main():
    root = Tk()
    root.geometry("310x310+400+100")
    app = Paint(root)
    root.mainloop()


if __name__ == '__main__':
    main()
mnist.predict(script_dir + '/my_drawing.png')
