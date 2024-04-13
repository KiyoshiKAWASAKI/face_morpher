"""
::

  Morph from source to destination face or
  Morph through all images in a folder

  Usage:
    morpher.py (--src=<src_path> --dest=<dest_path> | --images=<folder>)
              [--width=<width>] [--height=<height>]
              [--num=<num_frames>] [--fps=<frames_per_second>]
              [--out_frames=<folder>] [--out_video=<filename>]
              [--plot] [--background=(black|transparent|average)]

  Options:
    -h, --help              Show this screen.
    --src=<src_imgpath>     Filepath to source image (.jpg, .jpeg, .png)
    --dest=<dest_imgpath>   Filepath to destination image (.jpg, .jpeg, .png)
    --images=<folder>       Folderpath to images
    --width=<width>         Custom width of the images/video [default: 500]
    --height=<height>       Custom height of the images/video [default: 600]
    --num=<num_frames>      Number of morph frames [default: 20]
    --fps=<fps>             Number frames per second for the video [default: 10]
    --out_frames=<folder>   Folder path to save all image frames
    --out_video=<filename>  Filename to save a video
    --plot                  Flag to plot images to result.png [default: False]
    --background=<bg>       Background of images to be one of (black|transparent|average) [default: black]
    --version               Show version.
"""
from docopt import docopt
import os
import numpy as np
import cv2
import sys
from PIL import Image
import glob
# from tqdm import tqdm

# from facemorpher import locator
# from facemorpher import aligner
# from facemorpher import warper
# from facemorpher import blender
# from facemorpher import plotter
# from facemorpher import videoer
import locator
import aligner
import warper
import blender
import plotter
import videoer



def verify_args(args):
  if args['--images'] is None:
    valid = os.path.isfile(args['--src']) & os.path.isfile(args['--dest'])
    if not valid:
      print('--src=%s or --dest=%s file does not exist. Double check the supplied paths' % (
        args['--src'], args['--dest']))
      # exit(1)
      pass
  else:
    valid = os.path.isdir(args['--images'])
    if not valid:
      print('--images=%s is not a valid directory' % args['--images'])
      # exit(1)
      pass




def load_image_points(path, size):
  img = cv2.imread(path)
  points = locator.face_points(img)

  if len(points) == 0:
    print('No face in %s' % path)
    return None, None
  else:
    return aligner.resize_align(img, points, size)




def load_valid_image_points(imgpaths, size):
  for path in imgpaths:
    img, points = load_image_points(path, size)
    if img is not None:
      print(path)
      yield (img, points)





def list_imgpaths(images_folder=None, src_image=None, dest_image=None):
  if images_folder is None:
    yield src_image
    yield dest_image
  else:
    for fname in os.listdir(images_folder):
      if (fname.lower().endswith('.jpg') or
         fname.lower().endswith('.png') or
         fname.lower().endswith('.jpeg')):
        yield os.path.join(images_folder, fname)




def morph(src_img, src_points, dest_img, dest_points,
          video, width=500, height=600, num_frames=20, fps=10,
          out_frames=None, out_video=None, plot=False, background='black'):
  """
  Create a morph sequence from source to destination image

  :param src_img: ndarray source image
  :param src_points: source image array of x,y face points
  :param dest_img: ndarray destination image
  :param dest_points: destination image array of x,y face points
  :param video: facemorpher.videoer.Video object
  """
  size = (height, width)
  stall_frames = np.clip(int(fps*0.15), 1, fps)  # Show first & last longer
  plt = plotter.Plotter(plot, num_images=num_frames, out_folder=out_frames)
  num_frames -= (stall_frames * 2)  # No need to process src and dest image

  plt.plot_one(src_img)
  video.write(src_img, 1)

  # Produce morph frames!
  counter = 0

  for percent in np.linspace(1, 0, num=num_frames):
    points = locator.weighted_average_points(src_points, dest_points, percent)
    src_face = warper.warp_image(src_img, src_points, points, size)
    end_face = warper.warp_image(dest_img, dest_points, points, size)
    average_face = blender.weighted_average(src_face, end_face, percent)
    # print(average_face.shape)

    counter += 1

    if background in ('transparent', 'average'):
      mask = blender.mask_from_points(average_face.shape[:2], points)
      average_face = np.dstack((average_face, mask))

      if background == 'average':
        average_background = blender.weighted_average(src_img, dest_img, percent)
        average_face = blender.overlay_image(average_face, mask, average_background)

    # plt.plot_one(average_face)
    cv2.imwrite(out_video + "/morph_img_" + str(counter).zfill(4) + ".jpg", average_face)


    video.write(average_face)

  # plt.plot_one(dest_img)
  video.write(dest_img, stall_frames)
  # plt.show()





def morpher(imgpaths, width=500, height=600, num_frames=20, fps=10,
            out_frames=None, out_video=None, plot=False, background='black'):
  """
  Create a morph sequence from multiple images in imgpaths

  :param imgpaths: array or generator of image paths
  """
  out_video_final = os.path.join(out_video, "result.avi")
  video = videoer.Video(out_video_final, fps, width, height)
  images_points_gen = load_valid_image_points(imgpaths, (height, width))
  src_img, src_points = next(images_points_gen)

  for dest_img, dest_points in images_points_gen:
    morph(src_img, src_points, dest_img, dest_points, video,
          width, height, num_frames, fps, out_frames, out_video, plot, background)
    src_img, src_points = dest_img, dest_points

  video.end()




def main():
  # Define paths and parameters
  data_root_dir = "/afs/crc.nd.edu/group/cvrl/scratch_49/jhuang24/face_morph_v4_funneled"
  result_save_root_dir = "/afs/crc.nd.edu/group/cvrl/scratch_49/jhuang24/face_morph_v4_funneled_morphs"

  all_category = glob.glob(os.path.join(data_root_dir, '*'))
  print(all_category)

  # Loop through each category
  for one_cate in all_category:
      category_name = one_cate.split("/")[-1]

      all_people = glob.glob(os.path.join(one_cate, '*'))
      nb_morphs = int(len(all_people)/2)

      print("Current category: ", category_name)
      print("Number of morphs: ", nb_morphs)

      for i in range(nb_morphs):

          source_dir = all_people[i]
          target_dir = all_people[i + nb_morphs]

          print(source_dir, target_dir)

          # Generate morphs one by one
          one_source_img = glob.glob(os.path.join(source_dir, '*'))[0]
          one_target_img = glob.glob(os.path.join(target_dir, '*'))[0]

          print("Source Image: ", one_source_img)
          print("Target Image: ", one_target_img)

          target_save_dir = result_save_root_dir + "/" + category_name + "/" + \
                            source_dir.split("/")[-1] + "_to_" + target_dir.split("/")[-1]

          if not os.path.isdir(target_save_dir):
            os.mkdir(target_save_dir)
            print("Making directory: ", target_save_dir)

          morpher(imgpaths=list_imgpaths(None,
                                  one_source_img,
                                  one_target_img),
                    width=int(500),
                    height=int(600),
                    num_frames=int(150),
                    fps=int(30),
                    out_frames=None,
                    out_video=target_save_dir,
                    plot=False,
                    background="black")

          print("Finished generating one morph.")


if __name__ == "__main__":
  main()
