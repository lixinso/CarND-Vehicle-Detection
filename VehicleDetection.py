import numpy as np
import cv2
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from scipy.ndimage.measurements import label
import pickle
import glob
from random import shuffle
import os
import time
import math
from tqdm import tqdm
#from moviepy.editor import  VideoFileClip



vehicle_dirs = ["res/vehicles/GTI_Far/", "res/vehicles/GTI_Left/","res/vehicles/GTI_MiddleClose/","res/vehicles/GTI_Right/","res/vehicles/KITTI_extracted/"]
nonvehicle_dirs = ["res/non-vehicles/GTI/", "res/non-vehicles/Extras/"]

def list_vehicle_files(folder_list):

    files = []
    for dir in folder_list:
        filenames_short = os.listdir(dir )
        filenames = []
        for filename in filenames_short:
            if filename.endswith(".png"):
                filenames.append(dir + filename)

        #filenames = glob.glob(dir + "*.png")
        files += filenames

    return files

vehicles_files = list_vehicle_files(vehicle_dirs)
nonvehicles_files = list_vehicle_files(nonvehicle_dirs)




shuffle(vehicles_files)
shuffle(nonvehicles_files)

print("Number of     Vehicle Files:     ", len(vehicles_files))
print("Number of non-Vehicle File:       ", len(nonvehicles_files))

vehicle0      = mpimg.imread(vehicles_files[0])
vehicle0shape = vehicle0.shape
print("Shape of Image: ", vehicle0shape[0], vehicle0shape[1])


##Overview of the images
fig = plt.figure(figsize=(20,10))
for n in range(10):
    vehicle_idx = np.random.randint(0,len(vehicles_files))
    vehicle_image = mpimg.imread(vehicles_files[vehicle_idx])
    fig.add_subplot(1,10, n+1)
    plt.imshow(vehicle_image)
    plt.xticks(()); plt.yticks(());
    plt.title("Vehicle Image", fontsize=5)

    nonvehicle_idx = np.random.randint(0,len(nonvehicles_files))
    nonvehicle_image = mpimg.imread(nonvehicles_files[nonvehicle_idx])
    fig.add_subplot(2,10,n+1)
    plt.imshow(nonvehicle_image)
    plt.xticks(()); plt.yticks(());
    plt.title("Non-Vehicle Image", fontsize=5)
    n += 1

#plt.show()
plt.savefig("output_images/vehicle_non_vehicle_visualization.png", bbox_inches="tight")



#Visualize histogram of color in one image
def get_color_histogram(img, n_bins=32, bins_range=(0,256), plot=False):

    random_vehicle_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    hist_red   = np.histogram(random_vehicle_image[:,:,0], bins=32, range=(0,256))
    hist_green = np.histogram(random_vehicle_image[:,:,1], bins=32, range=(0,256))
    hist_blue  = np.histogram(random_vehicle_image[:,:,2], bins=32, range=(0,256))

    #hist_features = np.concatenate((hist_red[0], hist_green[0], hist_blue[0]))

    bin_edges = hist_red[1]

    xx=bin_edges[1:]
    yy=bin_edges[0:len(bin_edges)-1]
    xxyy = (xx + yy) /2
    bin_centers =(bin_edges[1:] + bin_edges[0:len(bin_edges)-1]) / 2

    hist_features = np.concatenate((hist_red[0], hist_green[0], hist_blue[0]))


    if plot == True:
        print("hist_features")

        fig = plt.figure(figsize=(4,4))
        matplotlib.rc('xtick', labelsize=20)
        matplotlib.rc('ytick', labelsize=20)
        plt.subplot(1,4,1)
        plt.imshow(random_vehicle_image)
        plt.title('Original Image:\n', fontsize=30)

        plt.subplot(1,4,2)
        plt.bar(bin_centers, hist_red[0], width=3)
        plt.xlim(0,256)
        plt.title("Red:\n", fontsize=30)

        plt.subplot(1,4,3)
        plt.bar(bin_centers, hist_green[0], width=3)
        plt.xlim(0,256)
        plt.title("Green:\n", fontsize=30)

        plt.subplot(1,4,4)
        plt.bar(bin_centers, hist_blue[0], width=3)
        plt.xlim(0,256)
        plt.title("Blue:\n", fontsize=30)

        #plt.subplots_adjust(left=0.5, right=2, top=1, bottom=0.)
        plt.savefig("output_images/histogram.png")
        #plt.show()

    return hist_red, hist_green, hist_blue, bin_centers, hist_features


vehicle_idx = np.random.randint(0,len(vehicles_files))
random_vehicle_image = cv2.imread(vehicles_files[vehicle_idx])
#hist_red_test, hist_green_test, hist_blue_test, bin_centers_test, hist_features_test = get_color_histogram(random_vehicle_image, True)


##Visualize the distribution of color in and image
def visualize_distribution_of_color():
    scale = max(random_vehicle_image.shape[0], random_vehicle_image.shape[1], 64) / 64
    img_small = cv2.resize(random_vehicle_image, (np.int(random_vehicle_image.shape[1] / scale ), np.int(random_vehicle_image.shape[0])), interpolation=cv2.INTER_NEAREST)
    img_small_RGB = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)
    img_small_rgb = img_small_RGB / 255

    f = plt.figure(figsize=(20,10))
    ax1 = f.add_subplot(1,2,1)
    f.tight_layout()
    matplotlib.rc('xtick', labelsize=40)
    matplotlib.rc('ytick', labelsize=40)
    ax1.imshow(random_vehicle_image)
    ax1.set_title('Original Image:\n', fontsize=40)

    ax2 = f.add_subplot(1,2,2, projection='3d')
    #ax2.text2D(0.15, 0.99, 'Image Color Distribution:\n', transform=ax2.transAxes, fontsize=60)
    ax2.set_xlim((0,255))
    ax2.set_ylim((0,255))
    ax2.set_zlim((0,255))

    ax2.tick_params(axis='both', which='major', labelsize=25, pad=8)
    ax2.scatter(img_small_RGB[:,:,0].ravel(), img_small_RGB[:,:,1].ravel(), img_small_RGB[:,:,2].ravel(), c=img_small_rgb.reshape((-1,3)), edgecolors='none')

    plt.show()

    plt.savefig("output_images/3d_color_distribution.png")

#visualize_distribution_of_color()

def image_spatial_binning(img_orig, size=(32, 32), display=False):
    color_space = 'RGB'
    size = (8,8)
    small_img_8x8 = cv2.resize(img_orig,size)

    if color_space == 'RGB':
        feature_img = np.copy(img_orig)
    elif color_space == 'HSV':
        feature_img = cv2.cvtColor(img_orig, cv2.COLOR_RGB2HSV)
    elif color_space == 'HLS':
        feature_img = cv2.cvtColor(img_orig, cv2.COLOR_RGB2HLS)
    elif color_space == 'YUV':
        feature_img = cv2.cvtColor(img_orig, cv2.COLOR_RGB2YUV)
    elif color_space == 'YCrCb':
        feature_img = cv2.cvtColor(img_orig, cv2.COLOR_RGB2YCrCb)

    feature_img_resize = cv2.resize(feature_img, size)
    features = feature_img_resize.ravel()


    if display == True:

        f = plt.figure(figsize=(3,3))
        ax1 = f.add_subplot(1,3,1)
        f.tight_layout()
        matplotlib.rc('xtick', labelsize=20)
        matplotlib.rc('ytick', labelsize=20)
        ax1.imshow(img_orig)
        ax1.set_title('Original Image:\n', fontsize=20)

        ax2 = f.add_subplot(1,3,2)
        ax2.imshow(small_img_8x8)
        ax2.set_title('8*8 Image\n', fontsize=20)

        ax3 = f.add_subplot(1,3,3)
        ax3.plot(features)

        plt.show()

        plt.savefig("output_images/spatial_binning.png")


    return features

#spatial_binning_test = image_spatial_binning(random_vehicle_image, display=True)

def get_hog_features(img):
    orientation = 9
    pixels_per_cell = (8,8)
    cells_per_block = (2,2)
    visualization=False
    feature_vector=True

    if visualization == True:
        features, hog_image = hog(img, orientations=orientation, pixels_per_cell=pixels_per_cell,
                                  cells_per_block=cells_per_block, transform_sqrt=False,
                                  visualise=True, feature_vector=False)

        return features, hog_image

    else:
        features = hog(img, orientations=orientation, pixels_per_cell=pixels_per_cell,
                       cells_per_block=cells_per_block,transform_sqrt=False,
                       visualise=False, feature_vector=feature_vector)
        return features

hog_feature_test = get_hog_features(random_vehicle_image[:,:,0])

def extract_features_one_image(image):
    file_features = []

    color_space = 'LUV'

    if color_space == 'RGB':
        feature_image = np.copy(image)
    elif color_space == 'HSV':
        feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    elif color_space == 'LUV':
        feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)

    ##spatial feature
    # spatial_features = get_bin_spatial(feature_image, size =(32,32) )
    spatial_features = image_spatial_binning(feature_image)

    file_features.append(spatial_features)

    ##hist features
    a1, a2, a3, a3, hist_features = get_color_histogram(feature_image, n_bins=32)
    file_features.append(hist_features)

    ##hog features
    # hog channel "ALL"
    hog_features = []
    orientations = 9
    pixels_per_cell = (8, 8)
    cells_per_block = (4,4)#(2, 2)
    for channel in range(feature_image.shape[2]):
        # hog_features_x = get_hog_features(feature_image[:,:,channel],orientations, pixels_per_cell, cells_per_block, visualization=False, feature_vector=True )
        hog_features_x = get_hog_features(feature_image[:, :, channel])

        hog_features.extend(hog_features_x)

    hog_features = np.ravel(hog_features)

    file_features.append(hog_features)

    return np.concatenate(file_features)

def extract_features(images):
    features = []
    for file in tqdm(images):
        img = mpimg.imread(file)
        concatenate_file_features = extract_features_one_image(img)
        features.append(concatenate_file_features)

    return features

from sklearn.externals import joblib
def train_svc_model():
    car_features = extract_features(vehicles_files)
    non_car_features = extract_features(nonvehicles_files)

    X = np.vstack((car_features, non_car_features)).astype(np.float64)
    X_scaler = StandardScaler().fit(X)
    scaled_X = X_scaler.transform(X)

    y = np.hstack((np.ones(len(car_features)), np.zeros(len(non_car_features))))

    X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=33)

    svc = LinearSVC()
    svc.fit(X_train, y_train)

    test_accuracy = svc.score(X_test, y_test)
    print("Test accuracy = ", test_accuracy)

    predicted = svc.predict(X_test[0:10])
    print(predicted)
    print(y_test[0:10])


    joblib.dump(svc, 'svcdump.pkl')
    joblib.dump(X_scaler, 'scaler.pkl')

#Parameters

#train_svc_model()


svc = joblib.load('svcdump.pkl')
X_scaler = joblib.load('scaler.pkl')
#predicted = svc.predict(X_test[0:10])
#print(predicted)


def sliding_windows(img, x_start_stop=[None, None], y_start_stop=[None,None], xy_window=(64,64), xy_overlap=(0.5,0.5)):

    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = int(img.shape[0] * 0.5)
    if y_start_stop[1]  == None:
        y_start_stop[1] = int(img.shape[1] * 0.9)


    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    nx_pix_per_step = np.int(xy_window[0] * (1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1] * (1 - xy_overlap[1]))
    nx_windows = np.int(xspan / nx_pix_per_step) -1
    ny_windows = np.int(yspan / ny_pix_per_step) -1
    window_list = []

    for ys in range(ny_windows):
        for xs in range(nx_windows):
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx   = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy   = starty + xy_window[1]
            window_list.append(((startx,starty),(endx, endy)))

    return window_list



def draw_boxes(img, bboxes, color=(0,0,255), thickness=6):
    img_copy = np.copy(img)
    for bbox in bboxes:
        cv2.rectangle(img_copy, bbox[0], bbox[1], color, thickness)
    return img_copy

def search_window(img, windows, svc, scaler):
    on_windows = []
    for window in windows:
        if window[1][1] < img.shape[0] and window[1][0] < img.shape[1]:
            test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]: window[1][0]], (64, 64))
            features = extract_features_one_image(test_img)
            features_reshaped = np.array(features).reshape(1,-1)
            test_features = scaler.transform(features_reshaped)
            prediction = svc.predict(test_features)
            #val = svc.decision_function(test_features)
            #print(val)
            #if val > -2:
            if prediction == 1:
                on_windows.append(window)

    return on_windows


#plt.clf()
test_img = mpimg.imread("test_images/test4.jpg")
#test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
draw_img = np.copy(test_img)


#test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
#windows = sliding_windows(test_img) #, xy_overlap=(0.85,0.85)
#print(windows)
#window_img = draw_boxes(test_img, windows)

#plt.savefig("output_images/slidding_window.png")


#selected_windows = search_window(test_img, windows, svc, X_scaler)
#window_img = draw_boxes(test_img, selected_windows)

#plt.imshow(window_img)
#plt.show()
#print(selected_windows)


##heatmap
def add_heat(heatmap, boxes):
    for box in boxes:
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    #heatmap[heatmap <= 4] = 0
    heatmap[heatmap <= 2] = 0
    return heatmap

def draw_heat(img, heatmap):
    labels = label(heatmap)
    print("hello")

    boxes = []
    for vehicle in range(1, labels[1]+1):
        nonzero = (labels[0] == vehicle).nonzero()
        nonzerox = np.array(nonzero[0])
        nonzeroy = np.array(nonzero[1])
        box = ( (np.min(nonzeroy), np.min(nonzerox)), (np.max(nonzeroy), np.max(nonzerox)))
        boxes.append(box)

    img = draw_boxes(img, boxes)
    return img

#heatmap = np.zeros_like(test_img[:,:,0]).astype(np.float)
#heatmap = add_heat(heatmap, windows)
#heatmap = np.clip(heatmap-2, 0, 255)

#np.set_printoptions(threshold=np.nan)

def video_pipeline(test_img, draw=False):
    test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
    backup_img = np.copy(test_img)
    test_img = test_img.astype(np.float32) / 255
    #test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
    windows = sliding_windows(test_img, xy_overlap=(0.85,0.85))  # ,
    selected_windows = search_window(test_img, windows, svc, X_scaler)
    window_img = draw_boxes(test_img, selected_windows)
    #np.set_printoptions(threshold=np.nan)
    heatmap = np.zeros_like(test_img[:, :, 0]).astype(np.float)
    heatmap = add_heat(heatmap, selected_windows)
    print(heatmap)
    print("\n\n\n")
    #heatmap = np.clip(heatmap - 2, 0, 255)
    heatmap = np.clip(heatmap, 0, 255)
    print(heatmap)
    print("\n\n\n")
    heat_image = draw_heat(backup_img, heatmap)

    #draw = True
    if draw:
        plt.clf()
        plt.imshow(heat_image)
        plt.show()

    return heat_image
    #return window_img

test_img = cv2.cvtColor(test_img, cv2.COLOR_RGB2BGR)
heatmap = video_pipeline(test_img, draw=True)
#plt.clf()
#plt.imshow(heatmap, cmap='hot')
#plt.imshow(heatmap)
#
#plt.imshow(heat_image)
#plt.show()


from moviepy.editor import VideoFileClip
output_video = "test_video_processed.mp4"
video_clip = VideoFileClip("test_video.mp4")
output_clip = video_clip.fl_image(video_pipeline)
output_clip.write_videofile(output_video,audio=False)

output_video = "project_video_processed.mp4"
video_clip = VideoFileClip("project_video.mp4")
output_clip = video_clip.fl_image(video_pipeline)
output_clip.write_videofile(output_video,audio=False)


