```
  Pupil Player Third Party Plugins by cpicanco
  Copyright (C) 2017 Rafael Picanço.

  Pupil Player is part of Pupil, a Pupil Labs (C) software, see <http://pupil-labs.com>.

  This program is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
```
# player_plugins

Run time plugins written to or modified from Pupil Player software (http://pupil-labs.com).

# overview

Please, note it is a work in progress (2018-02-23).

**Functional stuff as follows**:

- Main (export, analytics, functionality)
  - export_images
  - screen_tracker_offline
  - vis_circle_on_contours

- Dependencies
  - vcc_methods

- Visualization Only
  - display_recent_gaze_patch
  - filter_opencv_threshold

**Not functional/Experimental**
  - segmentation
  - kmeans_gaze_correction
  - trim_marks_patch
  - fixation_detector_patch

# install instructions

Just clone the repositories as follows:

```
# install git if necessary
# sudo apt install git

# clone dependencies
cd ~
git clone https://github.com/cpicanco/pupil_plugins_shared

# clone the repository
cd ~
mkdir pupil_player_settings
cd pupil_player_settings
git clone https://github.com/cpicanco/player_plugins.git plugins

```

# How to use the screen_tracker_offline plugin

Launch Pupil Player with the recording session you are interested in (you might want to download this [sample recording](https://drive.google.com/open?id=15iIx_QB6ZSg0FnWvcD_0XHU6C8Ea3s32)).

Then follow these steps:

a. Click on the 'Plugin Manager' icon on the right of the screen. The name of
the plugin ("Screen Tracker Offline") should appear in the plugin list. Click on
the corresponding circular icon (right of the name).

b. The plugin will open a small menu window. Click on the "Update Cache" button. The
plugin will scan all the video frames in your session to detect the corners of
the computer screen whenever present. Be patient, this may take a few minutes.
The screen will become shaded and freeze during detection. Once the detection is
over, the screen will brighten again, and you will be able to see the result of
the detection in the form of a blue trapezoid being superimposed on each frame
with the screen borders detected.

c. Click on the "Add screen surface" button. This step is required to add the
detected surface (in this case, the computer screen with its four corners) to the data.
This surface will appear as "Surface 0" with an arbitrary name, width,
and height. The width and height provided by default are 1 and 1, corresponding
to normalized 0-1 Cartesian coordinates for the screen surface. (However, you
are free to change these values for other ones, such as the width and height of
your computer monitor in pixels, for example.)

d. Finally, click on the Pupil button for exporting the data. (This is a
circular button on the left side of the screen with a download-like arrow.) The
data with the transformed coordinates will be saved in your session folder, in a
new subfolder created by Pupil and named after the time interval of the session
just processed.
