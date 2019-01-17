from tkinter import *

import numpy as np
import scipy.ndimage as ndimage
import skimage.io as ski_io
from skimage.transform import resize

import mnistApplication as nn


class Paint(object):
    FACTOR = 10
    DEFAULT_PEN_SIZE = FACTOR
    CANVAS_WIDTH = 28 * FACTOR
    CANVAS_HEIGHT = 28 * FACTOR

    def __init__(self):
        self.root = Tk()

        if __name__ == '__main__':
            self.nn = nn

        self.pen_button = Button(self.root, text='guess', command=self.make_guess)
        self.pen_button.grid(row=0, column=0)

        self.brush_button = Button(self.root, text='reset', command=self.reset_drawing)
        self.brush_button.grid(row=0, column=1)

        self.eraser_button = Button(self.root, text='pen', command=self.use_pen)
        self.eraser_button.grid(row=0, column=2)

        self.eraser_button = Button(self.root, text='eraser', command=self.use_eraser)
        self.eraser_button.grid(row=0, column=3)

        self.c = Canvas(self.root, bg='white', width=self.CANVAS_WIDTH, height=self.CANVAS_HEIGHT)
        self.c.grid(row=1, columnspan=5)

        self.setup()
        self.root.mainloop()

    def setup(self):
        self.old_x = None
        self.old_y = None
        self.line_width = self.DEFAULT_PEN_SIZE
        self.color = 'black'
        self.eraser_on = False
        self.active_button = self.pen_button
        self.c.bind('<B1-Motion>', self.paint)
        self.c.bind('<ButtonRelease-1>', self.reset)

    def make_guess(self):
        self.c.postscript(file="first_canvas.ps",
                          colormode="color",
                          width=self.CANVAS_WIDTH,
                          height=self.CANVAS_HEIGHT,
                          pagewidth=self.CANVAS_WIDTH - 1,
                          pageheight=self.CANVAS_HEIGHT - 1)
        data = ski_io.imread("first_canvas.ps")
        ski_io.imsave("canvas_image.jpg", data)
        loaded_image = ski_io.imread("canvas_image.jpg", as_gray=True)

        loaded_image = resize(loaded_image, (28, 28), anti_aliasing=True, anti_aliasing_sigma=0.5, mode='reflect')

        image_as_array = 1 - np.array(loaded_image).reshape(1, 28, 28).astype(int)
        image_as_array = 255 * image_as_array.astype(int)

        image_as_array = ndimage.gaussian_filter(image_as_array, sigma=0.8, order=0)

        image_as_array = image_as_array.reshape(1, 28, 28, 1)

        predict = nn.loaded_model.predict(image_as_array)
        print("I think it is a: " + str(np.argmax(predict[0])) + ", probability: " + str(predict[0][np.argmax(predict[0])]))


    def reset_drawing(self):
        self.c.delete(ALL)

    def use_pen(self):
        self.activate_button(self.pen_button)

    def use_brush(self):
        self.activate_button(self.brush_button)

    def use_eraser(self):
        self.activate_button(self.eraser_button, eraser_mode=True)

    def activate_button(self, some_button, eraser_mode=False):
        self.active_button.config(relief=RAISED)
        some_button.config(relief=SUNKEN)
        self.active_button = some_button
        self.eraser_on = eraser_mode

    def paint(self, event):
        self.line_width = self.DEFAULT_PEN_SIZE
        paint_color = 'white' if self.eraser_on else self.color
        if self.old_x and self.old_y:
            self.c.create_line(self.old_x, self.old_y, event.x, event.y,
                               width=self.line_width, fill=paint_color,
                               capstyle=ROUND, smooth=TRUE, splinesteps=36)
        self.old_x = event.x
        self.old_y = event.y

    def reset(self, event):
        self.old_x, self.old_y = None, None

if __name__ == '__main__':
    Paint()