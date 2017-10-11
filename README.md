```
  Pupil Player Third Party Plugins by cpicanco
  Copyright (C) 2016 Rafael Picanço.

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

Please, note it is a work in progress (2016-02-15).

**Functional stuff as follows**:

- Main (export, analytics, functionality)
  - export_images
  - fixation_detector_patch
  - screen_offline_detector
  - segmentation
  - trim_marks_patch
  - vis_circle_on_contours

- Dependencies
  - offline_reference_surface_patch
  - quad_segmentation
  - screen_detector
  - screen_detector_cacher
  - vcc_methods

- Visualization Only
  - display_recent_gaze_patch
  - filter_opencv_threshold

**Not functional yet**
  - kmeans_gaze_correction

# install instructions

Just clone the repositories as follows:

```
# clone dependencies
cd ~
git clone https://github.com/cpicanco/pupil_plugins_shared

# clone the repository
cd <pupil-folder>/player_settings/
git clone https://github.com/cpicanco/player_plugins.git plugins
```
