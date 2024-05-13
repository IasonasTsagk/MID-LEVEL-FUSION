#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import matplotlib.pyplot as plt
import numpy as np


image_l = cv2.imread(r"C:\Users\petsa\Downloads\left_p.jpg")
image_l = cv2.cvtColor(image_l, cv2.COLOR_BGR2RGB)
plt.imshow(image_l)


# In[2]:


import numpy as np
import cv2 as cv


#Convert to grayscale
gray_l = cv2.cvtColor(image_l, cv2.COLOR_BGR2GRAY).astype(np.uint8)
gray_l = np.float32(gray_l)

# Detect corners 
dst_l = cv2.cornerHarris(gray_l, 2, 3, 0.1)
dst_l = np.uint8(dst_l)

# Dilate corner image to enhance corner points
dst_l = cv2.dilate(dst_l,None)

plt.imshow(dst_l, cmap='gray')


# In[3]:


image_l[dst_l>0.01*dst_l.max()]=[252,210,15]
cv2.imwrite("highlighted_corners3.jpg", cv2.cvtColor(image_l, cv2.COLOR_RGB2BGR))


# In[4]:


image_l_copy = image_l.copy()  # Create a copy to draw squares on

# Threshold to obtain key-points
threshold = 0.01 * dst_l.max()
key_points_l = np.argwhere(dst_l > threshold)

# Define square size
square_size = 15

# Draw squares around each key-point
for point in key_points_l:
    y, x = point
    # Ensure the square doesn't go out of bounds
    if y - square_size >= 0 and y + square_size < gray_l.shape[0] and x - square_size >= 0 and x + square_size < gray_l.shape[1]:
        cv2.rectangle(image_l_copy, (x - square_size, y - square_size), (x + square_size, y + square_size), (252, 210, 15), 2)

# Save the modified image
cv2.imwrite("keypoints_with_squares5.jpg", cv2.cvtColor(image_l_copy, cv2.COLOR_RGB2BGR))

print("Image with keypoints and squares saved as 'keypoints_with_squares5.jpg'")


# In[5]:


import cv2
import matplotlib.pyplot as plt

image_r = cv2.imread(r"C:\Users\petsa\Downloads\right_p.jpg")
image_r = cv2.cvtColor(image_r, cv2.COLOR_BGR2RGB)
plt.imshow(image_r)
import numpy as np
import cv2 as cv


# Convert to grayscale
gray_r = cv2.cvtColor(image_r, cv2.COLOR_RGB2GRAY)
gray_r = np.float32(gray_r)

# Detect corners 
dst_r = cv2.cornerHarris(gray_r, 2, 3, 0.1)

# Dilate corner image to enhance corner points
dst_r = cv2.dilate(dst_r, None)
dst_r = np.uint8(dst_r)

plt.imshow(dst_r, cmap='gray')

image_r[dst_r > 0.01 * dst_r.max()] = [252, 210, 15]
cv2.imwrite("highlighted_corners5.jpg", cv2.cvtColor(image_r, cv2.COLOR_RGB2BGR))

image_r_copy = image_r.copy()  # Create a copy to draw squares on

# Threshold to obtain key-points
threshold = 0.01 * dst_r.max()
key_points_r = np.argwhere(dst_r > threshold)

# Define square size
square_size = 15

# Draw squares around each key-point
for point in key_points_r:
    y, x = point
    # Ensure the square doesn't go out of bounds
    if y - square_size >= 0 and y + square_size < gray_r.shape[0] and x - square_size >= 0 and x + square_size < gray_r.shape[1]:
        cv2.rectangle(image_r_copy, (x - square_size, y - square_size), (x + square_size, y + square_size), (252, 210, 15), 2)

# Save the modified image
cv2.imwrite("keypoints_with_squares5.jpg", cv2.cvtColor(image_r_copy, cv2.COLOR_RGB2BGR))

print("Image with keypoints and squares saved as 'keypoints_with_squares5.jpg'")


# In[6]:


# # Convert key-points to format suitable for SIFT
# key_points_l = [cv2.KeyPoint(float(x), float(y), 1) for y, x in key_points_l]
# key_points_r = [cv2.KeyPoint(float(x), float(y), 1) for y, x in key_points_r]


# # Create SIFT object
# sift = cv2.SIFT_create()


# # Compute SIFT descriptors
# key_points_l, descriptors_l = sift.detect(gray_l, key_points_l)
# key_points_r, descriptors_r = sift.detect(gray_r, key_points_r)

# # Now you have key-points and corresponding descriptors
# print("Number of SIFT keypoints(LEFT):", len(key_points_l))
# print("Shape of descriptors(LEFT):", descriptors_l.shape)
# # Now you have key-points and corresponding descriptors
# print("Number of SIFT keypoints(RIGHT):", len(key_points_r))
# print("Shape of descriptors(RIGHT):", descriptors_r.shape)


# In[7]:


sift = cv2.SIFT_create()
# Detect keypoints and compute descriptors for the left image
key_points_l, descriptors_l = sift.detectAndCompute(gray_l, None)

# Detect keypoints and compute descriptors for the right image
key_points_r, descriptors_r = sift.detectAndCompute(gray_r, None)


# In[28]:


import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import corner_harris, corner_peaks, corner_subpix, corner_orientations
from skimage.morphology import octagon

# Load the left and right images
image_l = cv2.imread(r"C:\Users\petsa\Downloads\left_p.jpg")
image_r = cv2.imread(r"C:\Users\petsa\Downloads\right_p.jpg")
image_l = cv2.cvtColor(image_l, cv2.COLOR_BGR2RGB)
image_r = cv2.cvtColor(image_r, cv2.COLOR_BGR2RGB)

# Convert left and right images to grayscale
gray_l = cv2.cvtColor(image_l, cv2.COLOR_RGB2GRAY)
gray_r = cv2.cvtColor(image_r, cv2.COLOR_RGB2GRAY)

# Detect corners for left image
points_l = corner_peaks(corner_harris(gray_l), min_distance=200, threshold_abs=0.001, threshold_rel=0.001)
print('{} corners were found in the left image.'.format(len(points_l)))

# Detect corners for right image
points_r = corner_peaks(corner_harris(gray_r), min_distance=200, threshold_abs=0.001, threshold_rel=0.001)
print('{} corners were found in the right image.'.format(len(points_r)))

# Create plot
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Plot left image with keypoints and squares
axes[0].imshow(image_l)
square_size = 15
for point in points_l:
    y, x = point
    if y - square_size >= 0 and y + square_size < gray_l.shape[0] and x - square_size >= 0 and x + square_size < gray_l.shape[1]:
        axes[0].plot([x - square_size, x + square_size, x + square_size, x - square_size, x - square_size], 
                     [y - square_size, y - square_size, y + square_size, y + square_size, y - square_size], 
                     color='yellow', linewidth=2)
axes[0].set_title('Left Image')
axes[0].axis('off')

# Plot right image with keypoints and squares
axes[1].imshow(image_r)
for point in points_r:
    y, x = point
    if y - square_size >= 0 and y + square_size < gray_r.shape[0] and x - square_size >= 0 and x + square_size < gray_r.shape[1]:
        axes[1].plot([x - square_size, x + square_size, x + square_size, x - square_size, x - square_size], 
                     [y - square_size, y - square_size, y + square_size, y + square_size, y - square_size], 
                     color='yellow', linewidth=2)
axes[1].set_title('Right Image')
axes[1].axis('off')

plt.show()


# In[13]:


def plot_images_with_keypoints(points_l, points_r):
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Plot left image with keypoints and squares
    axes[0].imshow(image_l)
    square_size = 15
    for point in points_l:
        y, x = point
        if y - square_size >= 0 and y + square_size < image_l.shape[0] and x - square_size >= 0 and x + square_size < image_l.shape[1]:
            axes[0].plot([x - square_size, x + square_size, x + square_size, x - square_size, x - square_size], 
                         [y - square_size, y - square_size, y + square_size, y + square_size, y - square_size], 
                         color='yellow', linewidth=2)
    axes[0].set_title('Left Image')
    axes[0].axis('off')

    # Plot right image with keypoints and squares
    axes[1].imshow(image_r)
    for point in points_r:
        y, x = point
        if y - square_size >= 0 and y + square_size < image_r.shape[0] and x - square_size >= 0 and x + square_size < image_r.shape[1]:
            axes[1].plot([x - square_size, x + square_size, x + square_size, x - square_size, x - square_size], 
                         [y - square_size, y - square_size, y + square_size, y + square_size, y - square_size], 
                         color='yellow', linewidth=2)
    axes[1].set_title('Right Image')
    axes[1].axis('off')

    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.show()




# In[14]:


# Detect corners for left image
points_l = corner_peaks(corner_harris(gray_l), min_distance=10, threshold_abs=0.01, threshold_rel=0.1)
print('{} corners were found in the left image.'.format(len(points_l)))

# Detect corners for right image
points_r = corner_peaks(corner_harris(gray_r), min_distance=10, threshold_abs=0.01, threshold_rel=0.1)
print('{} corners were found in the right image.'.format(len(points_r)))


plot_images_with_keypoints(points_l, points_r)


# In[15]:


# Detect corners for left image
points_l = corner_peaks(corner_harris(gray_l), min_distance=10, threshold_abs=0.01, threshold_rel=0.05)
print('{} corners were found in the left image.'.format(len(points_l)))

# Detect corners for right image
points_r = corner_peaks(corner_harris(gray_r), min_distance=10, threshold_abs=0.01, threshold_rel=0.05)
print('{} corners were found in the right image.'.format(len(points_r)))


plot_images_with_keypoints(points_l, points_r)


# In[16]:


# Detect corners for left image
points_l = corner_peaks(corner_harris(gray_l), min_distance=10, threshold_abs=0.005, threshold_rel=0.1)
print('{} corners were found in the left image.'.format(len(points_l)))

# Detect corners for right image
points_r = corner_peaks(corner_harris(gray_r), min_distance=10, threshold_abs=0.005, threshold_rel=0.1)
print('{} corners were found in the right image.'.format(len(points_r)))


plot_images_with_keypoints(points_l, points_r)


# In[17]:


# Detect corners for left image
points_l = corner_peaks(corner_harris(gray_l), min_distance=10, threshold_abs=0.005, threshold_rel=0.05)
print('{} corners were found in the left image.'.format(len(points_l)))

# Detect corners for right image
points_r = corner_peaks(corner_harris(gray_r), min_distance=10, threshold_abs=0.005, threshold_rel=0.05)
print('{} corners were found in the right image.'.format(len(points_r)))


plot_images_with_keypoints(points_l, points_r)


# In[18]:


# Detect corners for left image
points_l = corner_peaks(corner_harris(gray_l), min_distance=20, threshold_abs=0.01, threshold_rel=0.1)
print('{} corners were found in the left image.'.format(len(points_l)))

# Detect corners for right image
points_r = corner_peaks(corner_harris(gray_r), min_distance=20, threshold_abs=0.01, threshold_rel=0.1)
print('{} corners were found in the right image.'.format(len(points_r)))


plot_images_with_keypoints(points_l, points_r)


# In[19]:


# Detect corners for left image
points_l = corner_peaks(corner_harris(gray_l), min_distance=20, threshold_abs=0.01, threshold_rel=0.05)
print('{} corners were found in the left image.'.format(len(points_l)))

# Detect corners for right image
points_r = corner_peaks(corner_harris(gray_r), min_distance=20, threshold_abs=0.01, threshold_rel=0.05)
print('{} corners were found in the right image.'.format(len(points_r)))


plot_images_with_keypoints(points_l, points_r)


# In[20]:


# Detect corners for left image
points_l = corner_peaks(corner_harris(gray_l), min_distance=30, threshold_abs=0.01, threshold_rel=0.1)
print('{} corners were found in the left image.'.format(len(points_l)))

# Detect corners for right image
points_r = corner_peaks(corner_harris(gray_r), min_distance=30, threshold_abs=0.01, threshold_rel=0.1)
print('{} corners were found in the right image.'.format(len(points_r)))


plot_images_with_keypoints(points_l, points_r)


# In[21]:


# Detect corners for left image
points_l = corner_peaks(corner_harris(gray_l), min_distance=30, threshold_abs=0.01, threshold_rel=0.05)
print('{} corners were found in the left image.'.format(len(points_l)))

# Detect corners for right image
points_r = corner_peaks(corner_harris(gray_r), min_distance=30, threshold_abs=0.01, threshold_rel=0.05)
print('{} corners were found in the right image.'.format(len(points_r)))


plot_images_with_keypoints(points_l, points_r)


# In[64]:


# Detect corners for left image
points_l = corner_peaks(corner_harris(gray_l), min_distance=10, threshold_abs=0.01, threshold_rel=0.05)
print('{} corners were found in the left image.'.format(len(points_l)))
orientations_l = corner_orientations(gray_l, points_l, octagon(3,2))
orientations_l = np.rad2deg(orientations_l)


# Detect corners for right image
points_r = corner_peaks(corner_harris(gray_r), min_distance=10, threshold_abs=0.01, threshold_rel=0.05)
print('{} corners were found in the right image.'.format(len(points_r)))
orientations_r = corner_orientations(gray_r, points_r, octagon(3,2))
orientations_r = np.rad2deg(orientations_r)



# In[65]:


comb_l = []
comb_r = []

comb_l = [list(corner) + [orientation] for corner, orientation in zip(points_l, orientations_l)]
comb_r = [list(corner) + [orientation] for corner, orientation in zip(points_r, orientations_r)]
    


# In[66]:


comb_r


# In[72]:


patch_size = 25

# Transform the coordinates from Harris Corners function to Keypoint instances, so that we can use them with the SIFT function.
# Extract a patch around each of our detected corners' coordinates

kp_l = [cv2.KeyPoint(float(corner[1]), float(corner[0]), patch_size, angle = int(corner[2])) for corner in comb_l]
kp_r = [cv2.KeyPoint(float(corner[1]), float(corner[0]), patch_size, angle = int(corner[2])) for corner in comb_r]


# In[73]:


# fig, ((ax1),(ax2)) = plt.subplots(1, 2, figsize=(24, 12))

sift_l = cv2.drawKeypoints(image_l,kp_l,image_l,color=(0, 0, 255),flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
sift_r = cv2.drawKeypoints(image_r,kp_r,image_r,color=(0, 0, 255),flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
sift = cv2.SIFT_create()

# ax1.imshow(sift_l,cmap=plt.cm.gray)
# ax2.imshow(sift_r,cmap=plt.cm.gray)


# In[86]:


figure, ax = plt.subplots(1, 2, figsize=(24, 12))


kp_l, des_l = sift.detectAndCompute(gray_l,None)
img_l=cv2.drawKeypoints(gray_l,kp_l,image_l, flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
ax[0].imshow(img_l, cmap = "gray")

kp_r, des_r = sift.detectAndCompute(gray_r,None)
img_r=cv2.drawKeypoints(gray_r,kp_r,image_r, flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
ax[1].imshow(img_r, cmap = "gray")


# In[80]:


iterations = len(des_l) * len(des_r)
print("Number of iterations: ",iterations)


# In[87]:


def euclidian_distance(sift1,sift2):
    index_pairs = []
    eucl_dist = []
    ci=0
    for i in sift1:
        cj=0
        for j in sift2:
            index_pairs.append([ci,cj])
            norm_i = i / np.sqrt(np.sum(i**2))
            norm_j = j / np.sqrt(np.sum(j**2))
            eucl_dist.append(np.sqrt(sum((e1-e2)**2 for e1, e2 in zip(norm_i,norm_j))))
            cj=cj+1
        ci=ci+1
    return index_pairs, eucl_dist
            
index_pairs, eucl_dist = euclidian_distance(des_l,des_r)


# In[88]:


index_pairs = np.array(index_pairs)
eucl_dist = np.array(eucl_dist)
inds = eucl_dist.argsort()

# sorted_eucl_dist = np.flip(eucl_dist[inds])
# sorted_index_pairs = np.flip(index_pairs[inds])

sorted_eucl_dist = eucl_dist[inds]
sorted_index_pairs = index_pairs[inds]

# Results with the corresponding indices
print("Sorted euclidian distances (we show the 3 lowest):")
print(sorted_eucl_dist[:3])
print("\nSorted corner pairs based on their descriptor distances (we show the 3 best matches)")
print(sorted_index_pairs[:3])


# In[94]:


# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
# Match descriptors.
matches = bf.match(des_l,des_r)
# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)
# Draw first 10 matches.
img3 = cv2.drawMatches(image_l,kp_l,image_r,kp_r,matches[:10],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.imshow(img3),plt.show()


# In[1]:


good = []
for m,n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)


# In[95]:


kp_l = np.float32([kp.pt for kp in kp_l])
kp_r = np.float32([kp.pt for kp in kp_r])
H = None
if len(matches) > 4:
    ptsA = np.float32([kpsB[m.trainIdx] for m in matches])
    ptsB = np.float32([kpsA[m.queryIdx] for m in matches])
        
    (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,4) #threshold
print(H)


# In[ ]:





# In[97]:


new = cv2.warpPerspective(image_r, H, (image_r.shape[1] + image_l.shape[1], image_r.shape[0]))
new[0:image_l.shape[0], 0:image_l.shape[1]] = image_l
new = cv2.cvtColor(new, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(30, 20))
plt.imshow(new)


# In[ ]:




