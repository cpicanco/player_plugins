# -*- coding: utf-8 -*-
'''
  Copyright (C) 2015 Rafael Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
'''
import numpy as np

# modified from ~/pupil_src/capture/calibration_routines/calibrate.py

n_error_mess = "ERROR: Model n needs to be 3, 7, 9 or 15"

def calibrate_2d_polynomial(cal_pt_cloud,screen_size=(1280,720),threshold = 35, binocular=False, nu=15):
    """
    we do a simple two pass fitting to a pair of bi-variate polynomials
    return the function to map vector
    """
    # fit once using all avaiable data
    model_n = nu
    if binocular:
        model_n = 13

    cal_pt_cloud = np.array(cal_pt_cloud)

    cx,cy,err_x,err_y = fit_poly_surface(cal_pt_cloud,model_n)
    err_dist,err_mean,err_rms = fit_error_screen(err_x,err_y,screen_size)
    if cal_pt_cloud[err_dist<=threshold].shape[0]: #did not disregard all points..
        # fit again disregarding extreme outliers
        cx,cy,new_err_x,new_err_y = fit_poly_surface(cal_pt_cloud[err_dist<=threshold],model_n)
        map_fn = make_map_function(cx,cy,model_n)
        new_err_dist,new_err_mean,new_err_rms = fit_error_screen(new_err_x,new_err_y,screen_size)

        print 'first iteration. root-mean-square residuals: %s, in pixel'%err_rms
        print 'second iteration: ignoring outliers. root-mean-square residuals: %s in pixel'%new_err_rms

        print 'used %i data points out of the full dataset %i: subset is %i percent'\
            %(cal_pt_cloud[err_dist<=threshold].shape[0], cal_pt_cloud.shape[0], \
            100*float(cal_pt_cloud[err_dist<=threshold].shape[0])/cal_pt_cloud.shape[0])

        return map_fn,err_dist<=threshold,(cx,cy,model_n)

    else: # did disregard all points. The data cannot be represented by the model in a meaningful way:
        map_fn = make_map_function(cx,cy,model_n)
        print 'First iteration. root-mean-square residuals: %s in pixel, this is bad!'%err_rms
        print 'The data cannot be represented by the model in a meaningfull way.'
        return map_fn,err_dist<=threshold,(cx,cy,model_n)



def fit_poly_surface(cal_pt_cloud,n=7):
    M = make_model(cal_pt_cloud,n)
    U,w,Vt = np.linalg.svd(M[:,:n],full_matrices=0)
    V = Vt.transpose()
    Ut = U.transpose()
    pseudINV = np.dot(V, np.dot(np.diag(1/w), Ut))
    cx = np.dot(pseudINV, M[:,n])
    cy = np.dot(pseudINV, M[:,n+1])
    # compute model error in world screen units if screen_res specified
    err_x=(np.dot(M[:,:n],cx)-M[:,n])
    err_y=(np.dot(M[:,:n],cy)-M[:,n+1])
    return cx,cy,err_x,err_y

def make_model(cal_pt_cloud,n=7):
    n_points = cal_pt_cloud.shape[0]

    if n==3:
        X=cal_pt_cloud[:,0]
        Y=cal_pt_cloud[:,1]
        Ones=np.ones(n_points)
        ZX=cal_pt_cloud[:,2]
        ZY=cal_pt_cloud[:,3]
        M=np.array([X,Y,Ones,ZX,ZY]).transpose()

    elif n==7:
        X=cal_pt_cloud[:,0]
        Y=cal_pt_cloud[:,1]
        XX=X*X
        YY=Y*Y
        XY=X*Y
        XXYY=XX*YY
        Ones=np.ones(n_points)
        ZX=cal_pt_cloud[:,2]
        ZY=cal_pt_cloud[:,3]
        M=np.array([X,Y,XX,YY,XY,XXYY,Ones,ZX,ZY]).transpose()

    elif n==9:
        X=cal_pt_cloud[:,0]
        Y=cal_pt_cloud[:,1]
        XX=X*X
        YY=Y*Y
        XY=X*Y
        XXYY=XX*YY
        XXY=XX*Y
        YYX=YY*X
        Ones=np.ones(n_points)
        ZX=cal_pt_cloud[:,2]
        ZY=cal_pt_cloud[:,3]
        M=np.array([X,Y,XX,YY,XY,XXYY,XXY,YYX,Ones,ZX,ZY]).transpose()
    
    elif n==15:
        X=cal_pt_cloud[:,0]
        Y=cal_pt_cloud[:,1]

        XX=X*X
        YY=Y*Y
        XY=X*Y
        XXYY=XX*YY
        
        XXY=XX*Y
        YYX=YY*X

        XXXY=XX*XY
        YYXY=YY*XY

        XXXYY=XX*X*YY
        YYYXX=YY*Y*XX

        XXXXY=XX*XX*Y
        YYYYX=YY*YY*X

        Ones=np.ones(n_points)
        ZX=cal_pt_cloud[:,2]
        ZY=cal_pt_cloud[:,3]    
        M=np.array([X,Y,XX,YY,XY,XXYY,XXY,YYX,XXXY,YYXY,XXXYY,YYYXX,XXXXY,YYYYX,Ones,ZX,ZY]).transpose() 
    else:
        raise Exception(n_error_mess)
    return M


def make_map_function(cx,cy,n):
    if n==3:
        def fn((X,Y)):
            x2 = cx[0]*X + cx[1]*Y +cx[2]
            y2 = cy[0]*X + cy[1]*Y +cy[2]
            return x2,y2

    elif n==7:
        def fn((X,Y)):
            x2 = cx[0]*X + cx[1]*Y + cx[2]*X*X + cx[3]*Y*Y + cx[4]*X*Y + cx[5]*Y*Y*X*X +cx[6]
            y2 = cy[0]*X + cy[1]*Y + cy[2]*X*X + cy[3]*Y*Y + cy[4]*X*Y + cy[5]*Y*Y*X*X +cy[6]
            return x2,y2

    elif n==9:
        def fn((X,Y)):
            #          X         Y         XX         YY         XY         XXYY         XXY         YYX         Ones
            x2 = cx[0]*X + cx[1]*Y + cx[2]*X*X + cx[3]*Y*Y + cx[4]*X*Y + cx[5]*Y*Y*X*X + cx[6]*Y*X*X + cx[7]*Y*Y*X + cx[8]
            y2 = cy[0]*X + cy[1]*Y + cy[2]*X*X + cy[3]*Y*Y + cy[4]*X*Y + cy[5]*Y*Y*X*X + cy[6]*Y*X*X + cy[7]*Y*Y*X + cy[8]
            return x2,y2
    
    elif n==15:
        def fn((X,Y)):
            #          X,        Y,        XX,         YY,         XY,         XXYY,           XXY,          YYX,          XXXY,           YYXY,            XXXYY,             YYYXX,             XXXXY,             YYYYX,      Ones,
            x2 = cx[0]*X + cx[1]*Y + cx[2]*X*X + cx[3]*Y*Y + cx[4]*X*Y + cx[5]*Y*Y*X*X + cx[6]*Y*X*X + cx[7]*Y*Y*X + cx[8]*X*X*X*Y + cx[9]*Y*Y*X*Y + cx[10]*X*X*X*Y*Y + cx[11]*Y*Y*Y*X*X + cx[12]*X*X*X*X*Y + cx[13]*Y*Y*Y*Y*X + cx[14]
            y2 = cy[0]*X + cy[1]*Y + cy[2]*X*X + cy[3]*Y*Y + cy[4]*X*Y + cy[5]*Y*Y*X*X + cy[6]*Y*X*X + cy[7]*Y*Y*X + cy[8]*X*X*X*Y + cy[9]*Y*Y*X*Y + cy[10]*X*X*X*Y*Y + cy[11]*Y*Y*Y*X*X + cy[12]*X*X*X*X*Y + cy[13]*Y*Y*Y*Y*X + cy[14]
            return x2,y2
    else:
        raise Exception(n_error_mess)

    return fn

def fit_error_screen(err_x,err_y,(screen_x,screen_y)):
    err_x *= screen_x
    err_y *= screen_y
    err_dist=np.sqrt(err_x*err_x + err_y*err_y)
    err_mean=np.sum(err_dist)/len(err_dist)
    err_rms=np.sqrt(np.sum(err_dist*err_dist)/len(err_dist))
    return err_dist,err_mean,err_rms


if __name__ == '__main__':
    import cv2
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    from matplotlib import cm
    from mpl_toolkits.mplot3d import Axes3D

    from glob import glob
    import os
    import sys  

    reload(sys)  
    sys.setdefaultencoding('utf8')

    main_path = '/home/rafael/documents/doutorado/data_doc/003-Natan/2015-05-13/'
    cal_file = glob(os.path.join(main_path,'cal_pt_*'))[0]
    cal_pt_cloud = np.load(cal_file)
    # # plot input data
    # # Z = cal_pt_cloud
    # # ax.scatter(Z[:,0],Z[:,1],Z[:,2], c= "r")
    # # ax.scatter(Z[:,0],Z[:,1],Z[:,3], c= "b")

    # # fit once
    # model_n = 15

    # get_map_from_cloud()

    # cx,cy,err_x,err_y = fit_poly_surface(cal_pt_cloud,model_n)
    # map_fn = make_map_function(cx,cy,model_n)
    # err_dist,err_mean,err_rms = fit_error_screen(err_x,err_y,(1280,720))
    # print 'error_rms', err_rms,"in pixel"
    # threshold = 22 #22 #err_rms*2
    # print threshold 
    # # fit again disregarding crass outlines
    # cx,cy,new_err_x,new_err_y = fit_poly_surface(cal_pt_cloud[err_dist<=threshold],model_n)
    # map_fn = make_map_function(cx,cy,model_n)
    # new_err_dist,new_err_mean,new_err_rms = fit_error_screen(new_err_x,new_err_y,(1280,720))
    # print 'error_rms', new_err_rms,"in pixel"

    # print "using %i datapoints out of the full dataset %i: subset is %i percent" \
    #     %(cal_pt_cloud[err_dist<=threshold].shape[0], cal_pt_cloud.shape[0], \
    #     100*float(cal_pt_cloud[err_dist<=threshold].shape[0])/cal_pt_cloud.shape[0])

    # # plot residuals
    # fig_error = plt.figure()
    # plt.scatter(err_x,err_y,c="y")
    # plt.scatter(new_err_x,new_err_y)
    # plt.title("Resíduos dos dados brutos (amarelo) e subconjunto filtrado (azul)")

    # # load image file as numpy array
    img_folder ='export_images'
    path = os.path.join(main_path,img_folder)
    img_surface = cv2.imread(glob(os.path.join(path,'frame*'))[0],0)
    img_surface = cv2.cvtColor(img_surface, cv2.COLOR_GRAY2BGRA)
    img_surface[:,:,3] = 170

    # # plot projection of eye and world vs observed data
    # X,Y,ZX,ZY = cal_pt_cloud.transpose().copy()
    # X,Y = map_fn((X,Y))
    # X *= 1280
    # Y = 1-Y
    # Y *= 720
    # ZX *= 1280
    # ZY = 1-ZY
    # ZY *= 720

    # fig_projection = plt.figure()
    # surface = plt.imshow(img_surface)
    # plt.scatter(X,Y)
    # plt.scatter(ZX,ZY,c='y')
    # plt.ylim(ymax = 720, ymin = 0)
    # plt.xlim(xmax = 1280, xmin = 0)
    # plt.title("world space projection in pixes, mapped and observed (y)")


    # # plot the fitting functions 3D plot
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # outliers =cal_pt_cloud[err_dist>threshold]
    # inliers = cal_pt_cloud[err_dist<=threshold]
    # ax.scatter(outliers[:,0],outliers[:,1],outliers[:,2], c= "y")
    # ax.scatter(outliers[:,0],outliers[:,1],outliers[:,3], c= "y")
    # ax.scatter(inliers[:,0],inliers[:,1],inliers[:,2], c= "r")
    # ax.scatter(inliers[:,0],inliers[:,1],inliers[:,3], c= "b")
    # Z = cal_pt_cloud
    # X = np.linspace(min(Z[:,0]),max(Z[:,0]),num=30,endpoint=True)
    # Y = np.linspace(min(Z[:,1]),max(Z[:,1]),num=30,endpoint=True)
    # X, Y = np.meshgrid(X,Y)
    # ZX,ZY = map_fn((X,Y))
    # ZX,ZY
    # ax.plot_surface(X, Y, ZX, rstride=1, cstride=1, linewidth=.1, antialiased=True,alpha=0.4,color='r')
    # ax.plot_surface(X, Y, ZY, rstride=1, cstride=1, linewidth=.1, antialiased=True,alpha=0.4,color='b')
    # plt.xlabel("Pupil x in Eye-Space")
    # plt.ylabel("Pupil y Eye-Space")
    # plt.title("Z: Gaze x (blue) Gaze y (red) World-Space, yellow=outliers")

    # fig = plt.figure()
    # Z = cal_pt_cloud
    # ax = fig.gca(projection='3d')
    # ax.scatter(Z[:,0],Z[:,1],Z[:,2], c= "r")
    # ax.scatter(Z[:,0],Z[:,1],Z[:,3], c= "b")
    # X = np.linspace(min(Z[:,0]),max(Z[:,1]),num=20,endpoint=True)
    # Y = np.linspace(min(Z[:,0]),max(Z[:,1]),num=20,endpoint=True)
    # ZX = np.linspace(min(Z[:,2]),max(Z[:,2]),num=20,endpoint=True)
    # ZY = np.linspace(min(Z[:,3]),max(Z[:,3]),num=20,endpoint=True)
    # X, Y = np.meshgrid(X,Y)
    # ZX, ZY = np.meshgrid(ZX,ZY)
    # plt.title("raw samples")
    # ax.plot_surface(X,Y,ZX, rstride=1, cstride=1, linewidth=.1, antialiased=True,alpha=0.4,color='r')
    # ax.plot_surface(X,Y,ZY, rstride=1, cstride=1, linewidth=.1, antialiased=True,alpha=0.4,color='b')
    # # X,Y,_,_ = cal_pt_cloud.transpose()

    # # pts= map_fn((X,Y))
    # # import cv2
    # # pts = np.array(pts,dtype=np.float32).transpose()
    # # print cv2.convexHull(pts)[:,0]
    # plt.show()
    ns = [7]
    for nu in ns:
        map_fn,inlier_map,(cx,cy,model_n) = calibrate_2d_polynomial(cal_pt_cloud, nu=nu)
        # print cal_pt_cloud[inlier_map][:,0:2].shape
        # print cal_pt_cloud[inlier_map][0,2:4]
        #print map_fn, err_dist, cx, cy, model_n
        fn_input = cal_pt_cloud[:,0:2].transpose()
        cal_pt_cloud[:,0:2] =  np.array(map_fn(fn_input)).transpose()

        ref_pts = cal_pt_cloud[inlier_map][:,np.newaxis,2:4]
        ref_pts = np.array(ref_pts,dtype=np.float32)
        calib_bounds = cv2.convexHull(ref_pts)

        outliers = np.concatenate((cal_pt_cloud[~inlier_map][:,0:2],cal_pt_cloud[~inlier_map][:,2:4])).reshape(-1,2)
        inliers = np.concatenate((cal_pt_cloud[inlier_map][:,0:2],cal_pt_cloud[inlier_map][:,2:4]),axis=1).reshape(-1,2)
        
        figure, axes = plt.subplots()
        surface = plt.imshow(np.flipud(img_surface),cmap="Greys",origin='lower')
        # outliers
        X = outliers[:,0] 
        Y = outliers[:,1]
        X *= 1280
        Y *= 720
        axes.plot(X,Y,'r.')

        # inliers
        X = inliers[:,0] 
        Y = inliers[:,1]
        X *= 1280
        Y *= 720
        axes.plot(X,Y,'k.')
      
        X = calib_bounds[:,:,0]
        Y = calib_bounds[:,:,1]
        X *= 1280
        Y *= 720  
        
        axes.plot(X,Y,'g')
        plt.ylim(ymax = 720, ymin = 0)
        plt.xlim(xmax = 1280, xmin = 0)
        x_label = 'x (pixels)'
        y_label = 'y (pixels)'
        axes.set_xlabel(x_label)
        axes.set_ylabel(y_label)
        axes.xaxis.set_ticks_position('none')
        axes.yaxis.set_ticks_position('none') 
        axes.spines['top'].set_visible(False)
        axes.spines['bottom'].set_visible(False)
        axes.spines['left'].set_visible(False)
        axes.spines['right'].set_visible(False)
    plt.show()