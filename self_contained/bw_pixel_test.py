import numpy as np

my_pixels = [ # height
              [ # width
                [128, 0, 0, 0],[138, 0, 0, 0],[128, 134, 0, 0] # [b, g, r, a]
              ],
              [
                [200, 0, 0, 0],[200, 134, 0, 0],[200, 0, 128, 0]
              ],
              [
                [255, 0, 0, 0],[255, 0, 0, 0],[255, 134, 0, 128]
              ]
            ]

my_pixels = np.array(my_pixels)
# for height in my_pixels:
#   for pixel in height[:,:3]:
#     print np.mean(pixel)

test = my_pixels[:,:,0]
# test = my_pixels[:,:,1]
# test = my_pixels[:,:,2]
# test = my_pixels[:,:,3]

my_pixels[test == 128] = 0
print my_pixels 

  