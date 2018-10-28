import os
from pathlib import Path
from itertools import combinations, product
import sys
import mpl_toolkits.mplot3d.art3d as art3d
import numpy as np
import pandas as pd

from PIL import Image
from matplotlib import colors
from matplotlib import pyplot as plt
from matplotlib.patches import Wedge, Circle, Rectangle
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import mode
from skimage import color as skimage_color
from skimage import io, segmentation, util, exposure
from scipy.cluster.vq import vq, kmeans
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist
from multiprocessing import Pool, cpu_count
from itertools import chain
from webcolors import rgb_to_hex
import json


"""
GLOBAL VARIABLES
"""

color_terms = None
languages = ['de_short', 'de']
errors = []
b_w = []
base_dir = './results/'
verbose = True
do_json = False



def show_image(img):
    """
    Plots an Image as big as possible
    """

    width = 5.0
    height = img.shape[0] * width / img.shape[1]
    f = plt.figure(figsize=(width, height))
    plt.gca().set_axis_off()
    plt.imshow(img)


def plot_colors_3D(imagearray, ax, sampleQ=0.1, targetColorSpace='rgb', **kwargs):
    """
    Shows a 3D representation of colors in image array

    Parameters
    ----------
        imagearray = numpy array of RGB color values

        sampleQ = sample quota of pixels selected randomly, default = 0.1

    Return
    ------
        choice = array of randomly selected pixels without local reference
    """
    mask = np.random.choice([False, True], (imagearray.shape[0], imagearray.shape[1]), p=[1 - sampleQ, sampleQ])
    choice = imagearray[mask]
    if targetColorSpace == 'hsv':
        colors = choice.copy()
        choice = choice.reshape(choice.shape[0], -1, 3)
        choice = skimage_color.rgb2hsv(choice)
        plot_HSV_colors(choice, ax, colors, **kwargs)
    else:
        plot_RGB_colors(choice, ax, **kwargs)


def plot_RGB_colors(array, ax, pointsize=4, line_width=0):
    """
    Plots a 3D Color Distribution Graphic

    Parameters
    ----------
        array: numpy ndarray with RGB Color values
        pointsize: size of points, can be array (same length as x or y) or a number (default = 10)
    """
    # set axes' range
    ax.set_xlim3d(0, 255)
    ax.set_ylim3d(0, 255)
    ax.set_zlim3d(0, 255)

    # set axes' labels
    ax.set_xlabel('RED')
    ax.set_ylabel('GREEN')
    ax.set_zlabel('BLUE')

    # Set the background color of the pane YZ
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

    # get x, y and z values from image array (color channels)
    if len(array.shape) == 3:
        xValues = array[:, :, 0]
        yValues = array[:, :, 1]
        zValues = array[:, :, 2]
    else:
        xValues = array[:, 0]
        yValues = array[:, 1]
        zValues = array[:, 2]

    # facecolors in matplotlib scatter function requires RGB color values in [0...1] interval
    # reshaping array of color values to list for facecolor usage ( e.g. [ [0.1, 0.2, 0.3], [0.4, 0.5, 0.6], ... ] )
    # reshaping to linear Array
    if array.dtype == 'uint8':
        colorList = util.img_as_float(array).reshape(-1, 3).tolist()
    else:
        colorList = array.reshape(-1, 3).tolist()

    # if list of pointsizes given, values are normalized (0..1)
    if (type(pointsize) == list):
        pointsize = np.pi * (150 * np.array(_normalize(pointsize))) ** 2

    ax.scatter3D(xValues, yValues, zValues, s=pointsize, depthshade=False,
                 edgecolor='black', facecolors=colorList, linewidth=line_width)

    # draw cube
    r = [0, 255]
    for s, e in combinations(np.array(list(product(r, r, r))), 2):
        if np.sum(np.abs(s - e)) == r[1] - r[0]:
            ax.plot3D(*zip(s, e), color=(0.3, 0.3, 0.3), alpha=0.3)


def plot_HSV_colors(im_array, ax, colors, pointsize=15, radial=True):
    """
    Draws a 3D cylinder from HSV color image array

    Parameters:
    -----------
    im_array: Image array in HSV colors (to convert RGB to HSV use skimage.color.rgb2hsv)

    color_array: Array of colors for pixels in RGB!

    pointsize: a number or list of radius values for sizing the points

    radial: Boolean, if True plots a weel in z = 0 to distinguish angles

    """
    # Set the background color of the pane YZ
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

    # set axes' range
    ax.set_xlim3d(-1, 1)
    ax.set_ylim3d(-1, 1)
    ax.set_zlim3d(0, 1)

    if len(im_array.shape) == 3:
        xValues = im_array[:, :, 1] * np.sin(im_array[:, :, 0] * (2 * np.pi))
        yValues = im_array[:, :, 1] * np.cos(im_array[:, :, 0] * (2 * np.pi))
        zValues = im_array[:, :, 2]
    else:
        xValues = im_array[:, 1] * np.sin(im_array[:, 0] * (2 * np.pi))
        yValues = im_array[:, 1] * np.cos(im_array[:, 0] * (2 * np.pi))
        zValues = im_array[:, 2]

    colorList = util.img_as_float(colors)

    if radial:
        p1 = Circle((0, 0), 1, facecolor='none', linewidth=0.5, edgecolor=(0.3, 0.3, 0.3), alpha=0.3)
        p2 = Circle((0, 0), 1, facecolor='none', linewidth=0.5, edgecolor=(0.3, 0.3, 0.3), alpha=0.3)
        ax.add_patch(p1)
        ax.add_patch(p2)
        art3d.pathpatch_2d_to_3d(p1, z=1, zdir="z")
        art3d.pathpatch_2d_to_3d(p2, z=0, zdir="z")
        angles = np.linspace(0, np.pi * 5 / 6, 6)
        ys = np.cos(angles)
        xs = np.sin(angles)
        z_01 = np.array([0, 1])
        for i in range(xs.shape[0]):
            ax.plot([xs[i], -1 * xs[i]], [ys[i], -1 * ys[i]], zs=1, color=(0.3, 0.3, 0.3), alpha=0.3)
            ax.plot([xs[i], -1 * xs[i]], [ys[i], -1 * ys[i]], zs=0, color=(0.3, 0.3, 0.3), alpha=0.3)
            ax.plot([xs[i], xs[i]], [ys[i], ys[i]], zs=z_01, color=(0.3, 0.3, 0.3), alpha=0.3)
            ax.plot([-1 * xs[i], -1 * xs[i]], [-1 * ys[i], -1 * ys[i]], zs=z_01, color=(0.3, 0.3, 0.3), alpha=0.3)

    if colors.dtype == 'uint8':
        colorList = util.img_as_float(colors).reshape(-1, 3).tolist()

    # if list of pointsizes given, values are normalized (0..1)
    if type(pointsize) == list:
        pointsize = np.pi * (150 * np.array(_normalize(pointsize))) ** 2

    ax.scatter3D(xValues, yValues, zValues, s=pointsize, depthshade=False, alpha=1,
                 edgecolor='black', facecolors=colorList, linewidth=0)


def plot_CIELCH_colors(im_array, ax, colors, point_size=15, radial=False):
    """
    Draws a 3D cylinder from HSV color image array

    Parameters:
    -----------
    im_array: Image array in HSV colors (to convert RGB to HSV use skimage.color.rgb2hsv)

    color_array: Array of colors for pixels in RGB!

    point_size: a number or list of radius values for sizing the points

    radial: Boolean, if True plots a weel in z = 0 to distinguish angles

    """
    # Set the background color of the pane YZ
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

    if len(im_array.shape) == 3:
        xValues = im_array[:, :, 1] * np.sin(im_array[:, :, 2])
        yValues = im_array[:, :, 1] * np.cos(im_array[:, :, 2])
        zValues = im_array[:, :, 0]
    else:
        xValues = im_array[:, 1] * np.sin(im_array[:, 2])
        yValues = im_array[:, 1] * np.cos(im_array[:, 2])
        zValues = im_array[:, 0]

    colorList = util.img_as_float(colors)

    if radial:
        p1 = Circle((0, 0), 100, facecolor='none', linewidth=0.5, edgecolor=(0.3, 0.3, 0.3), alpha=0.3)
        p2 = Circle((0, 0), 100, facecolor='none', linewidth=0.5, edgecolor=(0.3, 0.3, 0.3), alpha=0.3)
        ax.add_patch(p1)
        ax.add_patch(p2)
        art3d.pathpatch_2d_to_3d(p1, z=1, zdir="z")
        art3d.pathpatch_2d_to_3d(p2, z=0, zdir="z")
        angles = np.linspace(0, np.pi * 5 / 6, 6)
        ys = np.cos(angles)
        xs = np.sin(angles)
        z_01 = np.array([0, 1])
        for i in range(xs.shape[0]):
            ax.plot([xs[i], -1 * xs[i]], [ys[i], -1 * ys[i]], zs=1, color=(0.3, 0.3, 0.3), alpha=0.3)
            ax.plot([xs[i], -1 * xs[i]], [ys[i], -1 * ys[i]], zs=0, color=(0.3, 0.3, 0.3), alpha=0.3)
            ax.plot([xs[i], xs[i]], [ys[i], ys[i]], zs=z_01, color=(0.3, 0.3, 0.3), alpha=0.3)
            ax.plot([-1 * xs[i], -1 * xs[i]], [-1 * ys[i], -1 * ys[i]], zs=z_01, color=(0.3, 0.3, 0.3), alpha=0.3)

    # if list of pointsizes given, values are normalized (0..1)
    if (type(point_size) == list):
        point_size = np.pi * (10 * np.array(_normalize(point_size))) ** 2

    ax.scatter3D(xValues, yValues, zValues, s=point_size, depthshade=False, alpha=1,
                 edgecolor='black', facecolors=colorList, linewidth=0)
    print(xValues.shape)
    print(point_size.shape)
    return np.stack((xValues, yValues, zValues, point_size.reshape(-1,1)), axis=-1)

def plot_CIE_Lab_colors(im_array, ax, point_size=15, **kwargs):
    """
    plots LAB colours in 3dimensional histogram using scatter 3D plotting
    """
    im_array, cnt = count_unique_rows(im_array)
    if im_array.shape[1] > 3:
        xValues = im_array[:,:,1]
        yValues = im_array[:,:,2]
        zValues = im_array[:,:,0]
        color_list = skimage_color.lab2rgb(im_array).reshape(-1,3).tolist()
    elif im_array.shape[1] == 3:
        xValues = im_array[:,1]
        yValues = im_array[:,2]
        zValues = im_array[:,0]
        color_list = skimage_color.lab2rgb(im_array.reshape(1,-1,3)).reshape(-1,3).tolist()
    
    if 'sizes' in kwargs:
        cnt = kwargs['sizes']
    if 'border' in kwargs:
        border = kwargs['border']
    else:
        border = 0

    cnt = np.pi * (point_size * np.array(_normalize(cnt))) ** 2

    ax.scatter3D(xValues, yValues, zValues, s=cnt, depthshade=False, alpha=1,
                 edgecolor='red', facecolors=color_list, linewidth=border)

def _normalize(lst):
    """
    helper function normalizing values and returns list of normalized values
    """
    s = max(lst)
    lst = map(lambda x: float(x) / s, lst)
    return list(lst)


def plot_segmented_colors(segmented_array, ax, colorSpace='rgb'):
    """
    Counts amount of pixels of corresponding segmented colors and plots 3D diagram

    Parameters
    ----------
    segmented_array: A numpy array containing RGB color rows after color segmentation

    ax : axis

    colorSpace: target color space, either 'rgb' or 'hsv'
    """

    seg_colors, color_counter = count_unique_rows(segmented_array)

    if colorSpace == 'rgb':
        if seg_colors.dtype is not 'float64':
            seg_colors = util.img_as_float(seg_colors.reshape(-1, 1, 3))
        plot_RGB_colors(seg_colors, ax, color_counter)
        return seg_colors
    elif colorSpace == 'hsv':
        colors = seg_colors.copy()
        seg_colors = seg_colors.reshape(seg_colors.shape[0], -1, 3)
        seg_colors = skimage_color.rgb2hsv(seg_colors)
        plot_HSV_colors(seg_colors, ax, colors, color_counter)
        return colors, color_counter
    elif colorSpace == 'LCh':
        colors = seg_colors.copy()
        seg_colors = seg_colors.reshape(-1, 1, 3)
        seg_colors = skimage_color.lab2lch(skimage_color.rgb2lab(seg_colors))
        return plot_CIELCH_colors(seg_colors, ax, colors, color_counter)


def _count_colors(segmented_colors):
    """
    helper function counting colour values on segmented_colors array
    """
    from collections import Counter
    colorList = []
    for color in segmented_array.reshape(-1, 3).tolist():
        colorList.append(tuple(color))

    countColor = Counter(colorList)

    seg_colors = np.array(list(countColor.keys()))
    color_counter = np.array(list(countColor.values()))
    return seg_colors, color_counter


def _plot_3D_color_histogramm(arr, ax):
    """
    ----------
    DEPRICATED
    ----------
    """
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax = Axes3D(fig)
    ax.set_xlim3d(0, 255)
    ax.set_ylim3d(0, 255)
    ax.set_zlim3d(0, 255)
    ax.set_xlabel('RED')
    ax.set_ylabel('GREEN')
    ax.set_zlabel('BLUE')
    norma = colors.Normalize(0, 255)
    colorArray = norma(arr).tolist()
    ax.scatter3D(arr[:, 0], arr[:, 1], arr[:, 2], facecolors=colorArray, depthshade=False, edgecolor='none')
    fig.show()


def plot_color_palette(color_array, ax, color_space='rgb'):
    """
    Plots a set of rectangulars from given RGB array.
    The array must have shape (n, 3)
    """

    # Format axes to plain style without ticks, etc.
    ax.xaxis.set_major_formatter(plt.NullFormatter())
    ax.yaxis.set_major_formatter(plt.NullFormatter())
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])

    # reshaping array to linear form ( [[r,g,b], [r,g,b], ... ] )
    color_array = color_array.reshape(-1, 3)

    colors = color_array.copy()
    if color_array.dtype == 'uint8':
        colors = util.img_as_float(colors)

    if color_space is 'lab':
        colors = skimage_color.lab2rgb(color_array.reshape(-1, 1, 3))

    # draws a rectangle for every row in array with corresponding color
    for idx, c in enumerate(colors):
        p = Rectangle((idx / len(color_array), 0), 1 / len(color_array), 0.3, color=c)
        ax.add_patch(p)


def plot_color_circle(colors, ret=False, input_space='hsv', **kwargs):
    """
    Draws a color circle from provided colors ordered by color hue.
    Colors align from 0 degree red and are then added to form a circle.
    Therefore the plotted angle doesn't correspond to hue angle!
    
    Parameters:
    -----------
    colors: numpy array (...[,...],3) in either hsv or rgb color model.
    
    ax: axis to plot on
    
    input_space: valid values: 'hsv' or 'rgb'
    
    """
    color_df = pd.DataFrame(colors.reshape(-1, 3))
    color_df = color_df.drop_duplicates()
    colors = color_df.as_matrix().reshape(1, -1, 3)
    uniq = colors.copy()

    if colors.dtype == 'uint8':
        colors = util.img_as_float(colors)
    if input_space == 'rgb':
        colors = skimage_color.rgb2hsv(colors).reshape(-1, 3)
    else:
        colors = colors.reshape(-1, 3)

    color_df = pd.DataFrame(colors)
    color_df = color_df.sort_values(0)
    colors = color_df.as_matrix()
    color_list = skimage_color.hsv2rgb(colors.reshape(1, -1, 3)).reshape(-1, 3).tolist()

    if 'ax' in kwargs:
        angle = 360 / len(color_list)

        for idx, color in enumerate(color_list):
            w = Wedge((0.5, 0.5), 0.5, (idx + 0.5) * angle, (idx + 1.5) * angle, width=0.250, linewidth=0,
                        color=color_list[idx])
            kwargs['ax'].add_patch(w)

        kwargs['ax'].set_axis_off()
    
    if ret:
        return color_list


def quickshift_segmentation(img_array, modus='n_most', ret=False, **kwargs):
    """
    Quickshif segmentation method with optional plotting (if ax defined)

    Parameters:
    -----------
    img_array : Image array (RGB)
    modus : kind of strategy assigning labels to coloured areas. 'n_most', 'mode' and 'avg' possible
    ret : Boolean, specifies if result should be returned
    ax (opt) : axis on where segmented image shall be plotted
    """
    
    if do_json:
        img = util.img_as_float(img_array)
    else:
        img = util.img_as_float(img_array[::2, ::2])
    segments = segmentation.quickshift(img, kernel_size=1, max_dist=20, ratio=0.1)

    if modus is 'avg':
        labels = skimage_color.label2rgb(segments, img, kind='avg')
    elif modus is 'mode':
        labels = label2rgb_mode(segments, img)
    else:
        labels = label2rgb_n_mode(segments, img)

    if 'ax' in kwargs:
        ax = kwargs['ax']
        ax.imshow(labels)
        ax.set_xlabel('# of Segments: %d' % len(np.unique(segments)))
        ax.set_axis_off()

    if ret:
        return labels


def octree_segmentation(img_array, ret=False, amount_colors=16, **kwargs):
    """
    Quantizes segmented Image to 'amount_colors' colors using Fast Octree Quantification algorithm

    Parameters
    ----------
    img_array : Image array (RGB)
    ret : Boolean, specifies if result should be returned
    amount_colors : amount of colours to be selected as upper quantification limit
    """
    if img_array.dtype != 'uint8':
        img_array = util.img_as_ubyte(img_array)

    segments = np.array(Image.fromarray(img_array).quantize(colors=amount_colors, method=2, kmeans=2).convert('RGB'))

    if 'ax' in kwargs:
        ax = kwargs['ax']
        ax.imshow(segments)
        ax.set_axis_off()

    if ret:
        return segments


def mediancut_segmentation(img_array, ret=False, amount_colors=16, **kwargs):
    """
    Quantizes segmented Image to 'amount_colors' colors using Median Cut Quantification algorithm

    Parameters
    ----------
    img_array : Image array (RGB)
    ret : Boolean, specifies if result should be returned
    amount_colors : amount of colours to be selected as quantification limit
    """
    if img_array.dtype != 'uint8':
        img_array = util.img_as_ubyte(img_array)

    segments = np.array(Image.fromarray(img_array).quantize(colors=amount_colors, method=0, kmeans=2).convert('RGB'))

    if 'ax' in kwargs:
        ax = kwargs['ax']
        ax.imshow(segments)
        ax.set_axis_off()

    if ret:
        return segments


def get_nearest_color(color, reference_colors, color_space='lab'):
    """
    Gets the argument and value of the nearest color in reference color array.
    The nearance is calculated with skimage.color.deltaE_ciede2000 function
    Parameters:
    -----------
    color: the color in lab color space to be compared
    reference_colors: array of lab colors; shape (..., ..., 3)
    Returns:
    --------
    arg = position of nearest color in reference colors
    nearest_color = nearest color in lab color space

    """
    color.reshape(-1, 1, 3)
    reference_colors.reshape(-1, 1, 3)

    if color_space is not 'lab':
        color = skimage_color.rgb2lab(color)
        # reference_colors = skimage_color.rgb2lab(reference_colors)

    arg = np.argmin(skimage_color.deltaE_ciede2000(color, reference_colors), axis=0)
    nearest_color = np.take(reference_colors, arg, axis=0)
    return arg, nearest_color


def set_global_terms():
    """
    DEPRICATED 
    """
    global color_terms
    terms_path = '/home/m/moosburger/Code/BA/terms/ral_colors.csv'
    color_terms = pd.read_csv(terms_path)

def closest_colors(color):
    """
    Calculates closest colour referred to global color_terms matrix
    returns closest color
    """
    return np.take(color_terms.as_matrix(), np.argmin(skimage_color.deltaE_ciede2000(color, color_terms[['l','a','b']].as_matrix()), axis=0), axis=0)

def closest_color_index(color):
    """
    returns index of closest color in global color_terms matrix
    """
    return np.argmin(skimage_color.deltaE_ciede2000(color, color_terms[['l','a','b']].as_matrix()), axis=0)

def get_color_term_mp(row):
    """
    DEPRICATED
    returns color terms from RAL matrix for row with l, a and b values
    """
    row['de'], row['en'], row['fr'], row['es'], row['it'], row['nl'], row['ral'], row['hex'], row['de_short'], row['en_short'], row['fr_short'] = closest_colors(np.array([row['l'], row['a'], row['b']]))[4:]
    return row

def get_color_term(color, color_term_map):
    """
    DEPRICATED
    """
    arg, near = get_nearest_color(color, color_term_map.iloc[:, 1:].as_matrix())
    return color_term_map.iloc[arg, 0]

def get_color_term_id(row):
    """
    takes row with l, a and b value and returns row with color_id (index of color terms table)
    """
    # index has to be increased by 1, since colour table starts with index 1 and closest color method returns index count from 0
    row['color_id'] = closest_color_index(np.array([row['l'], row['a'], row['b']])) + 1
    return row

def get_palette_terms(color_palette, color_term_map):
    """
    takes list of colors in color_palette and returns term list
    """
    color_list = []
    for color in color_palette:
        term = get_color_term(color, color_term_map)
        color_list.append(term)
    return color_list


def label2rgb_mode(label_field, image):
    """
    Visualise each segment in `label_field` with its mode color in `image`.
    Parameters
    ----------
    label_field : array of int
        A segmentation of an image.
    image : array, shape ``label_field.shape + (3,)``
        A color image of the same spatial shape as `label_field`.
    bg_label : int, optional
        A value in `label_field` to be treated as background.
    bg_color : 3-tuple of int, optional
        The color for the background label
    Returns
    -------
    out : array, same shape and type as `image`
        The output visualization.
    """
    out = np.zeros_like(image)
    labels = np.unique(label_field)

    for label in labels:
        mask = (label_field == label).nonzero()
        modi, _ = mode(image[mask], axis=0)
        color = modi[0, :]

        out[mask] = color
    return out


def label2rgb_n_mode(label_field, image, n_most=2):
    """
    Visualise each segment in `label_field` with its mode color in `image`.
    Parameters
    ----------
    label_field : array of int
        A segmentation of an image.
    image : array, shape ``label_field.shape + (3,)``
        A color image of the same spatial shape as `label_field`.
    bg_label : int, optional
        A value in `label_field` to be treated as background.
    bg_color : 3-tuple of int, optional
        The color for the background label
    Returns
    -------
    out : array, same shape and type as `image`
        The output visualization.
    """
    out = np.zeros_like(image)
    labels = np.unique(label_field)

    for label in labels:
        mask = (label_field == label).nonzero()
        color = _avg_n_most_frequent_color(image[mask].reshape(-1, 3), n_most=n_most)
        out[mask] = color
    return out


def count_unique_rows(arr, row_length=3):
    """
    Counts unique rows in 2+ array and returns unique rows and counter of rows

    Parameters:
    -----------
    arr : array with rows to be count
    row_length : default 3 for images, dimensions of rows
    
    Return:
    -------
    unique_rows, counter
    """
    arr = arr.reshape(-1, row_length)
    dt = np.dtype((np.void, arr.dtype.itemsize * arr.shape[1]))
    b = np.ascontiguousarray(arr).view(dt)
    unique_rows, row_counter = np.unique(b, return_counts=True)
    unique_rows = unique_rows.view(arr.dtype).reshape(-1, arr.shape[1])
    return unique_rows, row_counter


def _avg_n_most_frequent_color(arr, n_most=2):
    """
    Takes a :3 array and returns the average of the n most frequent colors weighted on frequency
    """
    uniq, cnt = count_unique_rows(arr)
    if cnt.shape[0] > n_most:
        freq_args = np.argpartition(cnt, -n_most)[-n_most:]
    else:
        freq_args = np.argpartition(cnt, -cnt.shape[0])[-cnt.shape[0]:]
    return np.average(uniq[freq_args], axis=0, weights=cnt[freq_args])


def palette2terms(color_palette, basic_terms, **kwargs):
    """
    Takes palette of colors and returns list of corresponding terms
    Optional plotting of terms if ax is given (matplotlib axes)
    """
    color_terms = []

    for color in color_palette:
        term = get_color_term(color, basic_terms)
        color_terms.append(term)

    if 'ax' in kwargs:
        ax = kwargs['ax']
        ax.set_title('Color Terms')
        color_text = ''
        for idx, term in enumerate(color_terms):
            color_text = color_text + term + ', '
            if (idx % 4 == 1):
                color_text += '\n'

        ax.text(0.2, 0.5, color_text)
        ax.set_axis_off()

    return set(color_terms)


def get_path(filename, folder, directory='/usr/local/home/'):
    """
    concats path from drectory, folder and filename
    """
    return directory + folder[5:] + filename


def get_terms_paths(dir):
    """
    creates list of paths on files in directory
    """
    paths = []
    for path in os.listdir(dir):
        paths.append(dir + path)

    return paths


def get_color_terms(path, terms, verbose=False):
    """
    DEPRICATED
    """
    try:
        img = io.imread(path)
    except OSError as e:
        return 'Error - probably file not found'
    if len(img.shape) is 2 or img.shape[2] is 1:
        return 'black & white'
    segmented_img = quickshift_segmentation(img, ret=True)  # , ax=ax2)
    quant_img = mediancut_segmentation(segmented_img, ret=True, amount_colors=16)
    color_palette = plot_color_circle(quant_img, ret=True, input_space='rgb')
    if verbose:
        print('Color terms for ' + path)
    return palette2terms(skimage_color.rgb2lab(color_palette.reshape(-1, 1, 3)), terms)


def match_image2terms(basic_term_paths, image_table, verbose, **kwargs): 
    """
    DEPRICATED
    """
    if 'image_path' in kwargs:
        directory = kwargs['image_path']
    else: 
        directory = '/usr/local/home/'

    image_table['path'] = image_table.apply(lambda row: get_path(row['filename'], row['folder'], directory), axis=1)
    if type(basic_term_paths) is list:
        basic_terms = []

        for terms_path in basic_term_paths:
            basic_terms.append(pd.read_csv(terms_path).set_index('index'))

        for idx, color_terms in enumerate(basic_terms):
            _, term_table_name = os.path.split(basic_term_paths[idx])
            image_table[term_table_name] = image_table.apply(lambda row: get_color_terms(row['path'], color_terms, verbose), axis=1)

    return image_table


def cluster_colors_k_means(color_array, n_colors=8, **kwargs):
    """
    takes an array of color values (NxMx3 recommended) and returns centroids of k-means clustering

    :param color_array: amount of centroids
    :param n_colors:
    :param kwargs:
    :return: array of centroids
    """
    centroids, _ = kmeans(color_array.reshape(-1,3), n_colors)
    pixel = np.reshape(color_array, (-1,3))
    qnt, _ = vq(pixel, centroids)

    if 'ax' in kwargs:
        centers_idx = np.reshape(qnt, (color_array.shape[0], color_array.shape[1]))    
        clustered = centroids[centers_idx]
        ax = kwargs['ax']
        ax.set_title('K-Means Quantization')
        ax.set_axis_off()
        ax.imshow(skimage_color.lab2rgb(clustered))

    return centroids, np.bincount(qnt)


def rescale_intensity(img, color_space='lab', ps_1=0, ps_2=97, pv_1=0, pv_2=98):
    """
    rescales intensity of saturation and brightness-channel of an image by converting into HSV color space

    :param img:
    :param color_space:
    :param ps_1:
    :param ps_2:
    :param pv_1:
    :param pv_2:
    :return:
    """
    if color_space is 'lab':
        img_hsv = skimage_color.rgb2hsv(skimage_color.lab2rgb(img))
    elif color_space is 'hsv':
        img_hsv = img
    else:
        img_hsv = skimage_color.rgb2hsv(img)

    ps_low, ps_high = np.percentile(img_hsv[:, :, 1], (ps_1, ps_2))
    img_hsv[:, :, 1] = exposure.rescale_intensity(img_hsv[:, :, 1], in_range=(ps_low, ps_high))

    pv_low, pv_high = np.percentile(img_hsv[:, :, 2], (pv_1, pv_2))
    img_hsv[:, :, 2] = exposure.rescale_intensity(img_hsv[:, :, 2], in_range=(pv_low, pv_high))

    if color_space is 'lab':
        return skimage_color.rgb2lab(skimage_color.hsv2rgb(img_hsv))
    elif color_space is 'hsv':
        return img_hsv
    else:
        return skimage_color.hsv2rgb(img_hsv)


def gamma_correction_sv(img, color_space='lab', s=0.7, v=0.65):
    """
    rescales intensity of saturation and brightness-channel of an image by converting into HSV color space

    :param img:
    :param color_space:
    :param s: gamma correction s channel (less is more staurated)
    :param v: gamma correction v channel (less is lighter)
    :return:
    """
    if color_space is 'lab':
        img_hsv = skimage_color.rgb2hsv(skimage_color.lab2rgb(img))
    elif color_space is 'hsv':
        img_hsv = img
    else:
        img_hsv = skimage_color.rgb2hsv(img)

    img_hsv[:, :, 1] = exposure.adjust_gamma(img_hsv[:, :, 1], s)
    img_hsv[:, :, 2] = exposure.adjust_gamma(img_hsv[:, :, 2], v)

    if color_space is 'lab':
        return skimage_color.rgb2lab(skimage_color.hsv2rgb(img_hsv))
    elif color_space is 'hsv':
        return img_hsv
    else:
        return skimage_color.hsv2rgb(img_hsv)


def get_contrasting_colors(color_array):
    """
    calculates distance matrix for colors in color_array
    returns:
    a : fist color of most distant
    b : second color of most distant
    max_dist : distance of these colors measured with deltaE_ciede2000 
    """
    distances = cdist(color_array, color_array, lambda a, b: skimage_color.deltaE_ciede2000(a,b))
    max_args = np.argmax(distances)
    t = np.unravel_index(max_args, distances.shape)
    return color_array[t[0]], color_array[t[1]], distances.reshape(-1)[max_args]


def get_term_from_color_mp(row):
    """
    gets color term for a given color by looking up a given color terms table

    Parameters:
    -----------
    idx : id of Image
    color: list of color dimensions (3) in L*a*b color space
    terms: pd.DataFrame of basic color terms and L*a*b values
    table: pd.DataFrame where result is stored
    
    """
    if type(languages) is not list:
        columns = [languages]
    else:
        columns = languages

    color_dict = {}

    for column in columns:
        color_dict[column] = []

    arg, near = get_nearest_color(np.array(color), terms[['l','a','b']].as_matrix())
    
    for column in columns:
        table.loc[table.shape[0]] = [idx, terms.loc[arg, column], column]
        
    print('terms added for image ', idx)


def get_palette_from_image(image, **kwargs):
    """
    segments color to color regions by applying quickshift segmentation and choosing mode colors
    then streches contrast on saturation and luminance color channel (HSV color space)
    finally clusters colors to 8 dominant colors and returns cluster centroids list as dominant colors
    """
    if 'json' in kwargs:
        do_json = kwargs['json']
    else:
        do_json = False
    segmented = quickshift_segmentation(image, ret=True, modus='n_most')
    segmented = skimage_color.rgb2lab(segmented)
    rescaled = gamma_correction_sv(segmented, s=0.8, v=0.9)
    if not np.isnan(rescaled).any():
        segmented = rescaled
    clustered_color_centroids, bin_counter = cluster_colors_k_means(segmented.reshape(-1,3), n_colors=16)
    clustered_hsv = skimage_color.rgb2hsv(skimage_color.lab2rgb(clustered_color_centroids.reshape(1,-1,3)))
    h_sorting = np.argsort(clustered_hsv.reshape(-1,3)[:,0])
    clustered_color_centroids = clustered_color_centroids[h_sorting]
    bin_counter = bin_counter[h_sorting]

    if clustered_color_centroids.shape[0] > 3:
        hull_lab = ConvexHull(clustered_color_centroids)
        clustered_hulled = clustered_color_centroids[hull_lab.vertices]
        bin_counter_hulled = bin_counter[hull_lab.vertices]
    else:
        hull_lab = -1
    
    if do_json:
        save_path = base_dir + 'json/' + kwargs['filename'] + "_segmented.jpg"
        img_to_save = Image.fromarray(np.uint8(skimage_color.lab2rgb(rescaled) *255))
        img_to_save.save(save_path, "JPEG", quality=80)

        unique_lab, counter_lab = count_unique_rows(segmented)
        colors_segm_rgb = skimage_color.lab2rgb(unique_lab.reshape(1,-1,3)) * 255
        uniq_segm_hex = np.apply_along_axis(rgb_to_hex, 1, colors_segm_rgb.astype(int).reshape(-1,3))

        colors_clust_rgb = skimage_color.lab2rgb(clustered_color_centroids.reshape(1,-1,3)) * 255
        uniq_clust_hex = np.apply_along_axis(rgb_to_hex, 1, colors_clust_rgb.astype(int).reshape(-1,3))

        if hull_lab == -1:
            vertics = True
        else:
            vertics = hull_lab.vertices

        json_object = _prepare_JSON_output(unique_lab, counter_lab, uniq_segm_hex, clustered_color_centroids, bin_counter, uniq_clust_hex, vertics)
        with open(base_dir + 'json/' + kwargs['filename'] + '.json', 'w') as outfile:
            json.dump(json_object, outfile)
        
        if verbose:
            print('created ', kwargs['filename'] + '.json')
    return clustered_hulled, bin_counter_hulled, hull_lab


def _prepare_JSON_output(segmented_lab, segmented_counter, segmented_hex, clustered_lab, clustered_counter, clustered_hex, hull_vertices):
    """
    takes clustered colors, sizes, hex-strings for segmented and clustered colors and prepares object for JSON object
    """
    json_object = {
        'segmented_colors': {
            'type': 'scatter3d',
            'mode': 'markers',
            'x': segmented_lab[:,1].tolist(),
            'y': segmented_lab[:,2].tolist(),
            'z': segmented_lab[:,0].tolist(),          
            'marker': {
                'sizemode': 'area',
                'line': {
                    'width': 0
                },
                'opacity': 1,
                'sizeref': 0.25,
                'color': segmented_hex.tolist(),
                'size': segmented_counter.tolist()
            }
        },
        'clustered_colors': {
            'type': 'scatter3d',
            'mode': 'markers',
            'showlegend': False,
            'x': clustered_lab[:,1].tolist(),
            'y': clustered_lab[:,2].tolist(),
            'z': clustered_lab[:,0].tolist(),          
            'marker': {
                'sizemode': 'area',
                'line': {
                    'width': 0
                },
                'opacity': 1,
                'sizeref': 1,
                'color': clustered_hex.tolist(),
                'size': clustered_counter.tolist()
            }
        },
        'convex_hull': {
            'type': 'scatter3d',
            'mode': 'markers',
            'showlegend': False,
            'x': clustered_lab[hull_vertices][:,1].tolist(),
            'y': clustered_lab[hull_vertices][:,2].tolist(),
            'z': clustered_lab[hull_vertices][:,0].tolist(),
            'marker': {
                'symbol': 'circle-open',
                'sizemode': 'area',
                'line': {
                    'width': 0,
                },
                'color': '#000',
                'color_pie': clustered_hex[hull_vertices].tolist(),
                'opacity': 1,
                'sizeref': 1,
                'size': clustered_counter[hull_vertices].tolist()
            }
        }
    }
    return json_object


def _add_palette_mp(row):
    """
    reads image and adds its 8 dominant colors to palette_table

    Parameters:
    -----------
    row
    """
    global b_w
    global errors
    global counter
    global metric

    try:
        if do_json:
            image = io.imread(row['url'] + row['path'])[::2, ::2] 
        else:
            image = io.imread(row['url'] + row['path'])[::2, ::2]
    except :
        e = sys.exc_info()[0]
        print('Error on image ',  row['index'], row['path'], e)
        row['error'] = str(e)
        return row
        
    if len(image.shape) is 2 or image.shape[2] is 1:
        if verbose:
            print('black & white image ', row['index'], row['path'])
        row['b/w'] = True
        return row

    if do_json:
        palette, bin_count, hull = get_palette_from_image(image, json=True, index=row['index'], filename=os.path.splitext(row['path'])[0])
    else:
        palette, bin_count, hull = get_palette_from_image(image)
    row['palette'] = list(palette)
    row['bin_count'] = list(bin_count)

    if hull != -1:
        row['volume'] = hull.volume
        _, _, row['distance'] = get_contrasting_colors(palette)
    else: 
        row['volume'] = 0
        row['distance'] = 0

    if verbose:  
        print(row)

    return row


def _add_jsons(row):
    """
    reads image and adds its 8 dominant colors to palette_table

    Parameters:
    -----------
    row
    """
    global b_w
    global errors
    global counter
    global metric

    try:
        image = io.imread(row['url'] + row['path'])
        
    except:
        e = sys.exc_info()[0]
        print('Error on image ',  row['index'], row['path'], e)
        row['error'] = str(e)
        return row
        
    if len(image.shape) is 2 or image.shape[2] is 1:
        if verbose:
            print('black & white image ', row['index'], row['path'])
        row['b/w'] = True
        return row

    palette, bin_count, hull = get_palette_from_image(image, json=True, index=row['index'], filename=os.path.splitext(row['path'])[0])
    row['palette'] = list(palette)
    row['bin_count'] = list(bin_count)

    if hull != -1:
        row['volume'] = hull.volume
        _, _, row['distance'] = get_contrasting_colors(palette)
    else: 
        row['volume'] = 0
        row['distance'] = 0

    if verbose:  
        print(row['index'], a, sep=',')
    return row
  

def run_mp(images, terms, output, json_):
    """
    Main method running on multiple threads
    calculates color extraction and calls color term method finally
    palette and metrics csv are written here
    """

    global color_terms 
    global base_dir
    global do_json

    do_json = json_
    base_dir = output
    color_terms = pd.read_csv(terms)

    os.makedirs(os.path.dirname(output), exist_ok=True)
    if do_json:
        os.makedirs(os.path.dirname(output + 'json/'), exist_ok=True)

    # take every image in images tables and save dominant colors (color palette) in image_colors table
    #palette_func = bake_in_func(_add_palette_mp)
    image_colors = parallelize_dataframe(images, apply_get_color_palette)

    # creates copy of dataframe with row for every list item in column 'result'
    # choose chain_from_iter instead to ignore cascading lists
    if 'b/w' in image_colors.columns:
        bw_df = image_colors.dropna(subset=['b/w'])
        image_colors.drop('b/w', axis=1)
    else:
        bw_df = pd.DataFrame()

    if 'error' in image_colors.columns:
        error_df = image_colors.dropna(subset=['error'])
        image_colors.drop('error', axis=1)
    else:
        error_df = pd.DataFrame()

    metrics_df = image_colors[['index', 'volume', 'distance']]
    metrics_df = metrics_df.dropna()
    metrics_df['index'] = metrics_df['index'].astype(int)
    metrics_df = metrics_df.set_index('index')
    metrics_df.index.rename('art_id', inplace=True)


    image_colors = image_colors.dropna(subset=['palette'])
    image_colors = image_colors.drop(['volume', 'distance'], axis=1)

    # following lines expands the array to one line per color-palette
    image_colors = pd.DataFrame({ 
    'index': np.repeat(image_colors['index'].values, [len(x) for x in (chain(image_colors['palette']))]), 
    'color': list(chain.from_iterable(image_colors['palette'])), 
    'quantity' : list(chain.from_iterable(image_colors['bin_count']))
    })

    # fill additional columns with items in palette list in column 'result'
    image_colors[['l', 'a', 'b']] = image_colors['color'].apply(pd.Series)
    image_colors = image_colors.drop('color', axis=1)

    if not os.path.isdir(base_dir):
        os.mkdir(base_dir)

    # convert img_id to int (better csv saving result)
    image_colors['index'] = image_colors['index'].astype(int)
    image_colors['quantity'] = image_colors['quantity'].astype(int)

    # save dominant colors to CSV file on base_dir
    image_colors.to_csv(base_dir + 'palette_' + image_filename, index=False)

    metrics_df.to_csv(base_dir + 'metrics_' + image_filename, index=False)

    # go on with color terms
    color_terms_mp(image_colors, error_df, bw_df)


def color_terms_mp(image_colors, error_df, bw_df):
    """
    second main method mapping colors to terms (with method apply_get_terms)
    stores terms, errors and bw csv files.
    """

    image_colors = parallelize_dataframe(image_colors, apply_get_terms) 
    image_colors = image_colors.drop(['l','a','b','quantity'], axis=1)
    image_colors = image_colors.drop_duplicates()

    # save as CSV
    image_colors['index'] = image_colors['index'].astype(int)
    image_colors.to_csv( base_dir + 'terms_' + image_filename, index=False)

    # store error images in CSV file
    try:
        error_df = error_df[['error','index','path','url']]
    except(KeyError):
        pass
    error_df.to_csv(base_dir + 'errors_' + image_filename, index=False)

    # store greyscale images in CSV file
    try:
        bw_df = bw_df[['bw','index']]
    except(KeyError):
        pass
    bw_df.to_csv(base_dir + 'bw_' + image_filename, index=False)
   
    

def start(input, inputmode, terms, output, verbose_, json_):
    """
    First method called
    prepares input and calls run_mp method
    """
    global image_filename
    global verbose

    verbose = verbose_

    if output[-1] != '/':
        output += '/'

    if inputmode == 'dir':
        if input[-1] != '/':
            input += '/'
        image_filename = Path(input).parts[-1] + '.csv'
        file_list = np.array([s for s in os.listdir(input) if s.endswith('.jpg')]).reshape(-1,1)
        images = pd.DataFrame(file_list, columns=['path'])
        images['url'] = input
        images['index'] = images.index.values
        images.index.rename('index', inplace=True)
        
        if verbose:
            print(images, '\nProcessing images ...')
        run_mp(images, terms, output, json_)

    else:
        # image_filename will be name base of result csv    
        image_filename = os.path.basename(input)
        images = pd.read_csv(input, dtype={'id': np.int64})
        run_mp(images, terms, output, json_)


def start_dir(dir_path, terms_path):
    """
    start calculations for images in dir
    """
    images = pd.DataFrame(columns=['id', 'path', 'url'])

    for filename in os.listdir(dir_path):
        if os.path.splitext(filename)[1] == '.jpg':
            images.loc[images.shape[0]] = [images.shape[0], filename, dir_path + '/']

    images['id'] = images['id'].astype(int)
    print(images)
    run(images, terms_path)


### Parallel Computing ###

def parallelize_dataframe(df, func):
    num_cores =  cpu_count() #number of cores on your machine
    num_partitions = num_cores * 4 #number of partitions to split dataframe

    df_split = np.array_split(df, num_partitions)
    pool = Pool(num_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df

def expand_result(row):
    row['result'] = list((row['one'], row['two'], row['three']))
    return row

def apply_get_color_palette(df):
    return df.apply(_add_palette_mp, axis=1)

def apply_get_terms(df):
    return df.apply(get_color_term_id, axis=1)








#########################   D E M O S   ###########################



def demo(path, terms, collection=True, quant_method='oct'):
    cterms = pd.read_csv(terms + '.csv')
    cterms.set_index('index')

    if collection:
        from skimage.io import ImageCollection
        img_set = ImageCollection(path)
        for img in img_set:
            _demo(img, cterms, quant_method)
    else:
        img = io.imread(path)
        _demo(img, cterms, quant_method)


def _demo(img, terms, quant_method):
    fig = plt.figure()
    fig.set_figheight(15)

    ax = fig.add_subplot(4, 2, 1)
    ax.set_axis_off()
    ax.set_title('Original Image')
    ax.imshow(img)

    ax1 = fig.add_subplot(4, 2, 2, projection='3d', aspect=1)
    ax1.set_title('RGB Color Space')
    plot_colors_3D(img, ax1)

    ax2 = fig.add_subplot(4, 2, 3)
    ax2.set_title('Quickshift Segmented')
    segmented_img = quickshift_segmentation(img, ret=True, ax=ax2)

    ax3 = fig.add_subplot(4, 2, 4, projection='3d', aspect=1)
    ax3.set_title('HSV Color Space')
    _ = plot_segmented_colors(segmented_img, ax3, colorSpace='LCh')

    ax4 = fig.add_subplot(4, 2, 5)
    if quant_method is 'oct':
        ax4.set_title('Octree Quantization')
        quant_img = octree_segmentation(segmented_img, ret=True, amount_colors=16, ax=ax4)
    elif quant_method is 'again':
        ax4.set_title('unweighted')
        quant_img = quickshift_segmentation(segmented_img, modus='mode', ret=True, ax=ax4)
    else:
        ax4.set_title('Median Cut Quantization')
        quant_img = mediancut_segmentation(segmented_img, ret=True, amount_colors=16, ax=ax4)

    ax5 = fig.add_subplot(4, 2, 6, aspect=1)
    ax5.set_title('Color Palette')
    color_palette = plot_color_circle(quant_img, ret=True, input_space='rgb', ax=ax5)

    ax6 = fig.add_subplot(4, 2, 7)
    palette2terms(skimage_color.rgb2lab(color_palette.reshape(-1, 1, 3)), terms, ax=ax6)


def demo20161214(dir, limit=10, db_path='/home/m/moosburger/BA/artigo_image_db.db', verbose=False, **kwargs):
    """
    Demo on Dec 14th 2016
    :param dir:
    :param limit:
    :param db_path:
    :param kwargs:
    :return:
    """
    disk_engine = create_engine('sqlite:///' + db_path)
    image_table = pd.read_sql_query('SELECT * FROM path ORDER BY RANDOM() LIMIT ' + str(limit), disk_engine)
    btp = get_terms_paths(dir)

    return match_image2terms(btp, image_table, verbose, **kwargs)


def demo20161230(path):
    """
    DEMO on Dec 30 2016
    :param path:
    :return:
    """
    image = skimage_color.rgb2lab(io.imread(path))
    segmented = quickshift_segmentation(image, ret=True, modus='n_most')
    segmented = rescale_intensity(segmented)
    clustered_color_centroids = cluster_colors_k_means(segmented, n_colors=8)
    return clustered_color_centroids