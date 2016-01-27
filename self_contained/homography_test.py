

if __name__ == '__main__':
    import cv2
    import numpy as np

    from quad_segmentation import sortCorners

    def m_verts_to_screen(verts, mapping):
        return cv2.getPerspectiveTransform(mapping,verts)

    def m_verts_from_screen(verts, mapping):
        return cv2.getPerspectiveTransform(verts,mapping)

    def detect_screens(gray_img, draw_contours=False):
        edges = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 17, -2)

        contours, hierarchy = cv2.findContours(edges,
                                        mode=cv2.RETR_TREE,
                                        method=cv2.CHAIN_APPROX_SIMPLE,offset=(0,0)) #TC89_KCOS
        
        if draw_contours:
            cv2.drawContours(gray_img, contours,-1, (0,0,0))
        
        # remove extra encapsulation
        hierarchy = hierarchy[0]
        contours = np.array(contours)

        # keep only contours                        with parents     and      children
        contours = contours[np.logical_and(hierarchy[:,3]>=0, hierarchy[:,2]>=0)]

        contours = np.array(contours)
        screens = []
        if contours != None: 
            # keep only > thresh_area   
            contours = [c for c in contours if cv2.contourArea(c) > (20 * 2500)]
            
            if len(contours) > 0: 
                # epsilon is a precision parameter, here we use 10% of the arc
                epsilon = cv2.arcLength(contours[0], True)*0.1

                # find the volatile vertices of the contour
                aprox_contours = [cv2.approxPolyDP(contours[0], epsilon, True)]

                # we want all contours to be counter clockwise oriented, we use convex hull for this:
                aprox_contours = [cv2.convexHull(c,clockwise=True) for c in aprox_contours if c.shape[0]==4]

                # a non convex quadrangle is not what we are looking for.
                rect_cand = [r for r in aprox_contours if r.shape[0]==4]
 
                # if draw_contours:
                #     cv2.drawContours(gray_img, rect_cand,-1, (0,0,0))

                # screens
                for r in rect_cand:
                    r = np.float32(r)

                    msg = 1

                    # define the criteria to stop and refine the screen verts
                    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
                    cv2.cornerSubPix(gray_img,r,(3,3),(-1,-1),criteria)

                    corners = np.array([r[0][0], r[1][0], r[2][0], r[3][0]])

                    # we need the centroid of the screen
                    M = cv2.moments(corners.reshape(-1,1,2))
                    centroid = np.array([M['m10']/M['m00'], M['m01']/M['m00']])
                    print 'a', centroid

                    # centroid = corners.sum(axis=0, dtype='float64')*0.25
                    # centroid.shape = (2)
                    # print 'b', centroid

                    # do not force dtype, use system default instead
                    # centroid = [0, 0]
                    # for i in corners:
                    #     centroid += i
                    # centroid *= (1. / len(corners))
                    # print 'c', centroid

                    # corners = sortCorners(corners, centroid)


                    # r[0][0], r[1][0], r[2][0], r[3][0] = corners[0], corners[1], corners[2], corners[3]

                    r_norm = r/np.float32((gray_img.shape[1],gray_img.shape[0]))
                    r_norm[:,:,1] = 1-r_norm[:,:,1]
                    screen = {'id':msg,'verts':r,'verts_norm':r_norm,'centroid':centroid,"frames_since_true_detection":0}
                    screens.append(screen)
        return screens

    real_shape = np.array([ 1280., 768. ])
    bias = np.array([ 41.60437012,  96.76702881])

    # 1) top,left
    # 2) bottom, left
    # 3) bottom, right
    # 4) top, right
    mapped_space_uv = np.array(((0,0),(1,0),(1,1),(0,1)),dtype=np.float32)
    mapped_space_to_screen = mapped_space_uv.reshape(-1,1,2)
    mapped_space_image = np.array(((0,0),(real_shape[0],0),(real_shape[0],real_shape[1]),(0,real_shape[1])),dtype=np.float32)


    #point = cv2.perspectiveTransform(bias, m_to_screen)

    # world_img = np.zeros((768,1280, 3), dtype=np.uint8)
    img_path = '/home/rafael/greserved/pupil-o/recordings/2015_05_27/cristiane/001/export_images/frame_1675.png'

    gray_world_img = cv2.imread(img_path, 0)

    detected_screens = detect_screens(gray_world_img, True)[0]
    s_vertices = detected_screens['verts']
    s_vertices = [v[0] for v in s_vertices]

    s_centroid = detected_screens['centroid']

    s_norm_vertices = np.array( detected_screens['verts_norm'] )
    s_norm_vertices.shape = (-1, 1, 2)
    m_to_screen = m_verts_to_screen(s_norm_vertices, mapped_space_uv)
    m_from_screen = m_verts_from_screen(s_norm_vertices, mapped_space_uv)

    m_point = np.append(s_centroid/real_shape, 1)

    p = np.dot(m_to_screen,m_point)
    p = p/p[2]
    p = np.delete(p, 2)
    print p
    p = real_shape*p

    # p = np.dot(m_from_screen, m_point)
    # print p/p[2]

    #s_uv = cv2.perspectiveTransform(s_norm_vertices,m_from_screen)
    #m_to_screen,_ = cv2.findHomography(srcPoints=s_uv,dstPoints=s_norm_vertices)
    #m_from_screen,_ = cv2.findHomography(srcPoints=s_norm_vertices,dstPoints=s_uv)

    for i,v in enumerate(s_vertices):
        cv2.circle(gray_world_img, (int(v[0]),int(v[1])), 3, (0, 0, 0), -1)
        cv2.putText(gray_world_img, str(i), (int(v[0] -10),int(v[1]) -10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 1, lineType = cv2.CV_AA )
    

    cv2.circle(gray_world_img, (int(s_centroid[0]),int(s_centroid[1])), 3, (0, 0, 0), 0)


    screen_space = cv2.perspectiveTransform(mapped_space_to_screen,m_to_screen).reshape(-1,2)

    #now we convert to image pixel coords
    screen_space[:,1] = 1-screen_space[:,1]
    screen_space[:,1] *= gray_world_img.shape[0]
    screen_space[:,0] *= gray_world_img.shape[1]

    #now we need to flip vertically again by setting the mapped_space verts accordingly.
    M = cv2.getPerspectiveTransform(screen_space,mapped_space_image)
    transformed_img = cv2.warpPerspective(gray_world_img,M, (int(real_shape[0]),int(real_shape[1])) )

    cv2.circle(transformed_img, (int(real_shape[0]/2),int(real_shape[1]/2)), 5, (0, 0, 0), -1)
    cv2.circle(transformed_img, (int(p[0]),int(p[1])),3, (0, 0, 0), -1)

    #cv2.namedWindow("output", cv2.CV_WINDOW_AUTOSIZE)
    while True:
        cv2.imshow("input", gray_world_img)
        cv2.imshow("output", transformed_img)
        ch = 0xFF & cv2.waitKey(1)
        if ch == 27:
            break
    cv2.destroyAllWindows()