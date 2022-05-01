import numpy as np
from PIL import Image

############### ---------- Basic Image Processing ------ ##############

### TODO 1: Read an Image and convert it into a floating point array with values between 0 and 1. You can assume a color image
def imread(filename):
    return np.array(Image.open(filename).convert('RGB'))/255.0

### TODO 2: Convolve an image (m x n x 3 or m x n) with a filter(l x k). Perform "same" filtering. Apply the filter to each channel if there are more than 1 channels


def convolve(img, filt):
    if len(img.shape)==2:
        # padding
        pad_t_b = int((filt.shape[0]-1)/2)
        pad_l_r = int((filt.shape[1]-1)/2)
  
        padded_img = np.pad(img, ((pad_t_b,pad_t_b),(pad_l_r,pad_l_r)), 'constant')
        
        # convolution
        output = np.zeros((img.shape[0],img.shape[1]))
        filter_flipped=np.flip(filt)
        for x in range(output.shape[0]):
          for y in range(output.shape[1]):
            output[x,y] = np.sum(filter_flipped * padded_img[x:x+filt.shape[0], y:y+filt.shape[1]])
        return output
    else:
        # padding
        pad_t_b = int((filt.shape[0]-1)/2)
        pad_l_r = int((filt.shape[1]-1)/2)
        
        padded_img = np.pad(img, ((pad_t_b,pad_t_b),(pad_l_r,pad_l_r),(0,0)), 'constant')
        
        # convolution
        output = np.zeros((img.shape[0],img.shape[1],img.shape[-1]))
        filter_flipped=np.flip(filt)
        for channel in range(output.shape[-1]):
          for x in range(output.shape[0]):
            for y in range(output.shape[1]):
              output[x,y,channel] = np.sum(filter_flipped * padded_img[x:x+filt.shape[0], y:y+filt.shape[1],channel])
        return output

### TODO 3: Create a gaussian filter of size k x k and with standard deviation sigma
def gaussian_filter(k, sigma):
    filt=np.zeros((k,k))
    mid=(k-1)/2
    for x in range(k):
      for y in range(k):
        dx=x-mid
        dy=y-mid
        exponent=(-dx*dx-dy*dy)/(2.0*sigma*sigma)
        filt[x,y]=np.exp(exponent)/(2.0*np.pi*sigma*sigma)
    sum_filt=np.sum(filt)
    filt=filt/sum_filt
    return filt
    
### TODO 4: Compute the image gradient. 
### First convert the image to grayscale by using the formula:
### Intensity = Y = 0.2125 R + 0.7154 G + 0.0721 B
### Then convolve with a 5x5 Gaussian with standard deviation 1 to smooth out noise. 
### Convolve with [0.5, 0, -0.5] to get the X derivative on each channel
### convolve with [[0.5],[0],[-0.5]] to get the Y derivative on each channel
### Return the gradient magnitude and the gradient orientation (use arctan2)
def gradient(img):
    #grayscale
    intensity_img=np.zeros((img.shape[0],img.shape[1]))
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            intensity_img[x,y]=0.2125*img[x,y,0]+0.7154*img[x,y,1]+0.0721*img[x,y,-1]
    #5x5 Gaussian
    filt1 = gaussian_filter(5,1)
    smooth_img=convolve(intensity_img,filt1)

    #[0.5, 0, -0.5]
    filt2=np.array([[0.5, 0, -0.5]])
    x_derivative=convolve(smooth_img,filt2)

    #[[0.5],[0],[-0.5]]
    filt3=np.array([[0.5],[0],[-0.5]])
    y_derivative=convolve(smooth_img,filt3)

    #gradient magnitude
    gradmag=np.sqrt(x_derivative*x_derivative+y_derivative*y_derivative)
    gradori=np.arctan2(y_derivative,x_derivative)
    return gradmag,gradori

##########----------------Line detection----------------

### TODO 5: Write a function to check the distance of a set of pixels from a line parametrized by theta and c. The equation of the line is:
### x cos(theta) + y sin(theta) + c = 0
### The input x and y are arrays representing the x and y coordinates of each pixel
### Return a boolean array that indicates True for pixels whose distance is less than the threshold
def check_distance_from_line(x, y, theta, c, thresh):
    boolean_pixels = np.zeros((x.size, y.size), dtype=bool)
    for (index_x, cur_x) in enumerate(x):
      for (index_y, cur_y) in enumerate(y):
        if np.absolute(cur_x*np.cos(theta)+cur_y*np.sin(theta)+c) < thresh: # slide 16 of linedetection
          # distance less than threshold
          boolean_pixels[index_y][index_x] = True
    return boolean_pixels

### TODO 6: Write a function to draw a set of lines on the image. The `lines` input is a list of (theta, c) pairs. Each line must appear as red on the final image
### where every pixel which is less than thresh units away from the line should be colored red
def draw_lines(img, lines, thresh):
    # img is height x width x RGB channels
    for [theta, c] in lines:
      should_color_red = check_distance_from_line(x=np.arange(img.shape[0]), y=np.arange(img.shape[1]), theta=theta, c=c, thresh=thresh)
      for x, _ in enumerate(img[:, 0, 0]):
        for y, _ in enumerate(img[0, :, 0]):
          if should_color_red[x][y]:
            img[x, y, :] = 0
            img[x, y, 0] = 1
    return img

### TODO 7: Do Hough voting. You get as input the gradient magnitude and the gradient orientation, as well as a set of possible theta values and a set of possible c
### values. If there are T entries in thetas and C entries in cs, the output should be a T x C array. Each pixel in the image should vote for (theta, c) if:
### (a) Its gradient magnitude is greater than thresh1
### (b) Its distance from the (theta, c) line is less than thresh2, and
### (c) The difference between theta and the pixel's gradient orientation is less than thresh3
# def hough_voting(gradmag, gradori, thetas, cs, thresh1, thresh2, thresh3):
#     #T x C array
#     pixels = np.zeros((len(thetas), len(cs)))
#     for x in range(gradmag.shape[0]):
#         for y in range(gradmag.shape[1]):
#             if gradmag[x,y]>thresh1:
#                 for taxis,theta in enumerate(thetas):
#                     for caxis,c in enumerate(cs):
#                         if (np.absolute(x*np.cos(theta)+y*np.sin(theta)+c) < thresh2) & (np.absolute(gradori[x,y]-theta)<thresh3):
#                             pixels[taxis,caxis]+=1
#                         else:
#                             continue
#             else:
#                 continue
#     return pixels

def hough_voting(gradmag, gradori, thetas, cs, thresh1, thresh2, thresh3):
  votes = np.zeros((thetas.size,cs.size), dtype=int)
  for ind_x, x in enumerate(gradmag[0, :]):
    # if ind_x % 100 == 1: print(ind_x)
    for ind_y, y in enumerate(gradmag[:, 0]):
      if gradmag[ind_y][ind_x] <= thresh1: continue
      for theta_index, theta in enumerate(thetas):
        if np.absolute(theta - gradori[ind_y][ind_x]) >= thresh3: continue
        for c_index, c in enumerate(cs):
          if np.absolute(ind_x*np.cos(theta)+ind_y*np.sin(theta)+c) < thresh2:
            votes[theta_index][c_index] +=1
  return votes   

### TODO 8: Find local maxima in the array of votes. A (theta, c) pair counts as a local maxima if (a) its votes are greater than thresh, and 
### (b) its value is the maximum in a nbhd x nbhd beighborhood in the votes array.
### Return a list of (theta, c) pairs
def localmax(votes, thetas, cs, thresh,nbhd):
  pad_num = int((nbhd-1)/2)
  padded_votes = np.pad(votes, pad_num, 'constant')
  local_maxima = []
  
  for col in range(pad_num, pad_num + votes.shape[0], 1):
    for row in range(pad_num, pad_num + votes.shape[1], 1):
      local_max = True
      if padded_votes[col][row] <= thresh:
        local_max = False
#         continue
      for i in range(-pad_num, pad_num):
        for j in range(-pad_num, pad_num):
          if padded_votes[col][row] < padded_votes[i+col][j+row]:
            local_max = False
      if local_max: 
        local_maxima.append((thetas[col-pad_num], cs[row-pad_num]))
  return local_maxima
  
# Final product: Identify lines using the Hough transform    
def do_hough_lines(filename):

    # Read image in
    img = imread(filename)

    # Compute gradient
    gradmag, gradori = gradient(img)

    # Possible theta and c values
    thetas = np.arange(-np.pi-np.pi/40, np.pi+np.pi/40, np.pi/40)
    imgdiagonal = np.sqrt(img.shape[0]**2 + img.shape[1]**2)
    cs = np.arange(-imgdiagonal, imgdiagonal, 0.5)

    # Perform Hough voting
    votes = hough_voting(gradmag, gradori, thetas, cs, 0.1, 0.5, np.pi/40)

    # Identify local maxima to get lines
    lines = localmax(votes, thetas, cs, 20, 11)

    # Visualize: draw lines on image
    result_img = draw_lines(img, lines, 0.5)

    # Return visualization and lines
    return result_img, lines
   
    
