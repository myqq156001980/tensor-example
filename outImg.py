# -*- coding: utf-8 -*-
from PIL import Image
import struct
"""
将MNIST数据转换为图片格式
"""

def read_image(filename, path):
    f = open(filename, 'rb')
    index = 0
    buf = f.read()
    f.close()
    magic, images, rows, colums = struct.unpack_from('>IIII', buf, index)
    index += struct.calcsize('>IIII')

    for i in range(images):
        image = Image.new('L', (colums, rows))

        for x in range(rows):
            for y in range(colums):
                image.putpixel((y, x), int(struct.unpack_from('>B', buf, index)[0]))
                index += struct.calcsize('>B')

        image.save(path + str(i) + '.png')
