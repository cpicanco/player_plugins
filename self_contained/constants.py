# -*- coding: utf-8 -*-
'''
  Copyright (C) 2016 Rafael Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
'''

from methods import get_pixels_per_degree, get_visual_angle

# object size, in cm
SCREEN_WIDTH = 70.0

# how far the object was from the participant's head, in cm
SCREEN_DISTANCE = 240.0

# size of the screen monitor, in pixels; used as real values of the screen surface 
SCREEN_WIDTH_PX, SCREEN_HEIGHT_PX = 1280,764

# physical measurements of the screen projection in degrees of the participant's visual angle
# SCREEN_WIDTH_DEG, SCREEN_HEIGHT_DEG = 15.3336085236, 9.15224758754
# SCREEN_WIDTH_DEG, SCREEN_HEIGHT_DEG = 16.5942899397, 9.90471680774
SCREEN_WIDTH_DEG, SCREEN_HEIGHT_DEG = get_visual_angle(SCREEN_WIDTH, SCREEN_DISTANCE)

PIXELS_PER_DEGREE = get_pixels_per_degree(SCREEN_WIDTH_PX, SCREEN_HEIGHT_PX, SCREEN_WIDTH_DEG, SCREEN_HEIGHT_DEG)