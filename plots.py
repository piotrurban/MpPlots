from time import sleep
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Circle, Rectangle, RegularPolygon
import random
import shutil
from pathlib import Path

def gen_circles():
    for i in range(0,100):
        sleep(0.2)
        fig, ax = plt.subplots()
        ax.set(xlim=(-200,200), ylim=(-200,200))
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        img1 = mpimg.imread("img1.jpg")
        plt.imshow(img1, extent=(80 ,120,0,20))
        shift = (random.randint(-100,100), random.randint(-100,100))
        circle = Circle(shift, 10 + 10*random.random()*5.0)
        circle.set_color('orange')
        ax.add_artist(circle)
        #plt.show()
        #plt.savefig("out_circle_" + str(i) + ".jpg")

def gen_squares():
    shutil.os.makedirs("out/train/square", 0o777, True)

    for i in range(0,100):
        sleep(0.1)
        fig, ax = plt.subplots()
        ax.set(xlim=(-200,200), ylim=(-200,200))
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        img1 = mpimg.imread("img1.jpg")
        plt.imshow(img1, extent=(80 ,120,0,20))
        shift = (random.randint(-100,100), random.randint(-100,100))
        side = random.randint(10,100)
        square = Rectangle(shift, side, side)
        ax.add_artist(square)
        plt.savefig("out/train/square/out_square_" + str(i) + ".jpg")

def gen_triangles(path: Path):
    shutil.os.makedirs(path, 0o777, True)
    for i in range(0,100):
        sleep(0.1)
        fig, ax = plt.subplots()
        ax.set(xlim=(-200,200), ylim=(-200,200))
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        img1 = mpimg.imread("img1.jpg")
        plt.imshow(img1, extent=(80 ,120,0,20))
        shift = (random.randint(-100,100), random.randint(-100,100))
        side = random.randint(10,100)
        triangle = RegularPolygon(xy=shift, numVertices=3, radius=side, orientation=random.random())
        ax.add_artist(triangle)
        plt.savefig(path/f"triangle_{i}.jpg")
        plt.close()

gen_triangles(Path("./testem/triangle"))
