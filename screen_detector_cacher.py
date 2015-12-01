# -*- coding: utf-8 -*-
'''
  Pupil Player Third Party Plugins by cpicanco
  Copyright (C) 2015 Rafael Picanço.

  Pupil Player is part of Pupil, a Pupil Labs (C) software, see <http://pupil-labs.com>.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
'''

# hacked from marker_detector_cacher

def fill_cache(visited_list,video_file_path,timestamps,q,seek_idx,run,min_marker_perimeter):
    '''
    this function is part of marker_detector it is run as a seperate process.
    it must be kept in a seperate file for namespace sanatisation
    '''
    import os
    import logging
    logger = logging.getLogger(__name__+' with pid: '+str(os.getpid()) )
    logger.debug('Started cacher process for Marker Detector')
    import cv2
    from video_capture import File_Capture, EndofVideoFileError,FileSeekError
    from screen_detector_methods import detect_markers_robust
    aperture = 9
    markers = []
    cap = File_Capture(video_file_path,timestamps=timestamps)

    def next_unvisited_idx(frame_idx):
        try:
            visited = visited_list[frame_idx]
        except IndexError:
            visited = True # trigger search

        if not visited:
            next_unvisited = frame_idx
        else:
            # find next unvisited site in the future
            try:
                next_unvisited = visited_list.index(False,frame_idx)
            except ValueError:
                # any thing in the past?
                try:
                    next_unvisited = visited_list.index(False,0,frame_idx)
                except ValueError:
                    #no unvisited sites left. Done!
                    logger.debug("Caching completed.")
                    next_unvisited = None
        return next_unvisited

    def handle_frame(nextf):
        if nextf != cap.get_frame_index():
            #we need to seek:
            logger.debug("Seeking to Frame %s" %nextf)
            try:
                cap.seek_to_frame(nextf)
            except FileSeekError:
                #could not seek to requested position
                logger.warning("Could not evaluate frame: %s."%nextf)
                visited_list[nextf] = True # this frame is now visited.
                q.put((nextf,[])) # we cannot look at the frame, report no detection
                return
            #seeking invalidates prev markers for the detector
            markers[:] = []

        try:
            frame = cap.get_frame_nowait()
        except EndofVideoFileError:
            logger.debug("Video File's last frame(s) not accesible")
             #could not read frame
            logger.warning("Could not evaluate frame: %s."%nextf)
            visited_list[nextf] = True # this frame is now visited.
            q.put((nextf,[])) # we cannot look at the frame, report no detection
            return

        ########################
        gray = cv2.cvtColor(frame.img,cv2.COLOR_BGR2GRAY)
        #######################
        markers[:] = detect_markers_robust(gray,
                                        grid_size = 5,
                                        prev_markers=markers,
                                        min_marker_perimeter=min_marker_perimeter,
                                        aperture=aperture,
                                        visualize=0,
                                        true_detect_every_frame=1)

        visited_list[frame.index] = True
        q.put((frame.index,markers[:])) #object passed will only be pickeled when collected from other process! need to make a copy ot avoid overwrite!!!

    while run.value:
        nextf = cap.get_frame_index()
        if seek_idx.value != -1:
            nextf = seek_idx.value
            seek_idx.value = -1
            logger.debug("User required seek. Marker caching at Frame: %s"%nextf)


        #check the visited list
        nextf = next_unvisited_idx(nextf)
        if nextf == None:
            #we are done here:
            break
        else:
            handle_frame(nextf)


    logger.debug("Closing Cacher Process")
    cap.close()
    q.close()
    return