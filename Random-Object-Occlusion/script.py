from PIL import Image
import random
import numpy as np
from matplotlib import pyplot as plt
import cv2
import os
import pdb
import sys
import math

directory = "/research/hal-datastage/datasets/processed/CelebA/celebahq_crop/all_images/imgs"
results = "/research/hal-datastage/datasets/processed/CelebA/celebahq_crop/all_images/random_occ"
results_seg = "/research/hal-datastage/datasets/processed/CelebA/celebahq_crop/all_images/random_occ_seg"

directory = "imgs"
results = "results"
results_seg = "results"

percent_of_area_of_face_covered_min, percent_of_area_of_face_covered_max = 15, 40
rotation_min, rotation_max = 0, 360

def get_segmentation_mask(object_png : Image.Image, image_to_occlude: Image.Image, placed_position):
    """[summary]

    Args:
        object_png (Image.Image): PIL Image of the object png that will be placed on the image to be occluded. It has already been scaled and rotated.
        image_to_occlude (Image.Image): [description]
        placed_position (tuple[int, int]): [description]

    Returns:
        [type]: [description]
    """
    image_to_occlude_width, image_to_occlude_height = image_to_occlude.size

    # First get all pixels that are not transparent from the png object image
    non_zero_alpha_r, non_zero_alpha_c = np.array(object_png)[:,:,3].nonzero()

    # Now translate
    non_zero_r = non_zero_alpha_r + placed_position[1]
    non_zero_c = non_zero_alpha_c + placed_position[0]

    # Seg mask will be same size as the image to be occluded
    segmentation_mask = np.zeros(image_to_occlude.size[::-1])
    segmentation_mask[non_zero_r[non_zero_r < image_to_occlude_height], non_zero_c[non_zero_c < image_to_occlude_width]] = 1

    return segmentation_mask

def detect_face_location(image_path: str):
    # save face detection algorithm's name as haarcascade
    haarcascade = "haarcascade_frontalface_alt2.xml"

    # pdb.set_trace()

    # create an instance of the Face Detection Cascade Classifier
    detector = cv2.CascadeClassifier(haarcascade)

    image_rgb = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    # Detect faces using the haarcascade classifier on the "grayscale image"
    faces = detector.detectMultiScale(cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY))

    return faces[0] if len(faces) > 0 else (0, 0, image_rgb.shape[1], image_rgb.shape[0])

if __name__ == "__main__":
    if len(sys.argv) > 1:
        directory = sys.argv[1]
        results = sys.argv[2]
        results_seg = sys.argv[3]

for file in os.listdir(directory):
    background = Image.open(f"{directory}/{file}")
    bg_width, bg_height = background.size

    scale = math.sqrt(random.randint(percent_of_area_of_face_covered_min, percent_of_area_of_face_covered_max) / 100)
    rotation = random.randint(rotation_min, rotation_max)

    face_x, face_y, face_width, face_height = detect_face_location(f"{directory}/{file}")
    location = (random.randint(face_x, face_x + face_width // 2), random.randint(face_y, face_y + face_height // 2))

    new_width = int(face_width * scale)
    new_height = int(face_height * (face_width * scale) // face_width)
    img = Image.open(f"Object Images/{random.choice(os.listdir('Object Images'))}").resize((new_width, new_height))

    img = img.rotate(rotation)

    background.paste(img, location, img)
    # background.show()
    background.save(f"{results}/{file[:-4]}-occluded.png")

    try:
        seg_mask = get_segmentation_mask(img, background, location)
    except:
        print(f"Error when computing seg mask for {file}")
        continue
    # overlap_img = Image.fromarray((0.5*np.array(background)*(1-seg_mask[:,:,None]) + 128*seg_mask[:,:,None].repeat(3, axis=2)).astype(np.uint8))
    # plt.imshow(seg_mask)
    # plt.show()

    cv2.imwrite(f"{results_seg}/{file[:-4]}-occluded-segmask.png", seg_mask * 255)
    print(f"Occluded {file}")
    # break