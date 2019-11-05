from pathlib import Path
from xml.etree.ElementTree import Element
from xml.etree.ElementTree import SubElement
from xml.etree.ElementTree import tostring

import click
import cv2
import numpy as np


class Source:
    """Class for image sources."""

    def __init__(self, path):
        self.path = Path(path)

    def images(self):
        """Generator function to yield cv2 image with filepath."""
        if self.path.is_file():
            yield (cv2.imread(self.path.as_posix()), self.path.absolute())
        else:
            for filename in self.path.iterdir():
                yield (cv2.imread(filename.as_posix()), filename.absolute())


def process(img_rec, show_steps, show_result):
    """Process image - find convex hulls of photocoagulation areas."""
    img, path = img_rec
    filename = path.stem
    if show_steps:
        cv2.imshow(f"{filename} - input image", img)
        cv2.waitKey(0)

    # 1. crop retina
    gray = cv2.medianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 5)
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=50, minRadius=50, maxRadius=0
    )
    circles = np.uint16(np.around(circles))
    mask = np.full((img.shape[0], img.shape[1]), 0, dtype=np.uint8)
    cv2.circle(mask, (circles[0, 0, 0], circles[0, 0, 1]), circles[0, 0, 2], (255, 255, 255), -1)
    cropped = cv2.bitwise_or(img, img, mask=mask)
    if show_steps:
        cv2.imshow(f"{filename} - cropped retina", cropped)
        cv2.waitKey(0)

    # 2. convert to L*a*b*
    img = cv2.cvtColor(cropped, cv2.COLOR_BGR2Lab)
    L, a, b = cv2.split(img)
    if show_steps:
        cv2.imshow(f"{filename} - luminance before", L)
        cv2.waitKey(0)
    # 3. averaging filter on Luminance channel
    avgL = cv2.blur(L, (3, 3))
    # 4. adaptive historgram equalization of Luminance
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    corrL = clahe.apply(avgL)
    if show_steps:
        cv2.imshow(f"{filename} - luminance after", corrL)
        cv2.waitKey(0)
    avgImg = cv2.merge((corrL, a, b))

    # 5. convert back to BGR
    img = cv2.cvtColor(avgImg, cv2.COLOR_Lab2BGR)
    b, g, r = cv2.split(img)
    # 6. median filtering of Green channel with 5x5 kernel
    medianG = cv2.medianBlur(g, 5)
    img = cv2.merge((b, medianG, r))
    if show_steps:
        cv2.imshow(f"{filename} - pre-processed", img)
        cv2.waitKey(0)

    # 7. detect laser marks
    gray2 = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    laser_marks = cv2.HoughCircles(
        gray2, cv2.HOUGH_GRADIENT, 1, 3, param1=50, param2=20, minRadius=0, maxRadius=10
    )
    if laser_marks is None or len(laser_marks[0]) < 2:
        return (None, path)  # no laser photocoagulation
    laser_marks = cv2.HoughCircles(
        gray2, cv2.HOUGH_GRADIENT, 1, 1, param1=40, param2=17, minRadius=2, maxRadius=10
    )
    marks = np.uint16(np.around(laser_marks))
    mask2 = np.full((img.shape[0], img.shape[1]), 255, dtype=np.uint8)
    for i in marks[0, :]:
        cv2.circle(mask2, (i[0], i[1]), i[2], (0, 0, 0), -1)

    masked = cv2.bitwise_or(img, img, mask=mask2)
    if show_steps:
        cv2.imshow(f"{filename} - laser marks", masked)
        cv2.waitKey(0)
    # 8. store laser marks for each quadrant
    q1 = np.empty((0, 2), int)
    q2 = np.empty((0, 2), int)
    q3 = np.empty((0, 2), int)
    q4 = np.empty((0, 2), int)
    for point in marks[:, :, 0:2][0]:
        if point[0] < img.shape[1] // 2:
            if point[1] < img.shape[0] // 2:
                q2 = np.append(q2, [point], axis=0)
            else:
                q3 = np.append(q3, [point], axis=0)
        else:  # q1, q4
            if point[1] < img.shape[0] // 2:
                q1 = np.append(q1, [point], axis=0)
            else:
                q4 = np.append(q4, [point], axis=0)
    # 9. find convex hull for each quadrant
    hull = []
    for i, q in enumerate((q1, q2, q3, q4)):
        if not q.size:
            hull.append(None)
            continue
        hull.append(cv2.convexHull(q.astype("float32")).astype(int))
        cv2.drawContours(masked, [hull[i]], 0, (255, 0, 0), 1, 8)

    if show_steps or show_result:
        cv2.imshow(f"{filename} - photocoaqulation areas", masked)
        cv2.waitKey(0)
    return (hull, path)


def save_xml(hull_rec_list, out):
    """Save list of hull records as XML."""
    xml = Element("photocoagulation_list")
    for hull, filepath in hull_rec_list:
        image = SubElement(xml, "image")
        path = SubElement(image, "path")
        path.text = filepath.as_posix()
        laser_marks = SubElement(image, "laser_marks")
        laser_marks.text = "False"
        if hull:
            laser_marks.text = "True"
            for i in range(len(hull)):
                area_i = SubElement(image, f"area{i}")
                area_i.text = " ".join([f"{x},{y}" for x, y in hull[i][:, 0]])
    with open(out, "wb") as f:
        f.write(tostring(xml))


@click.command()
@click.argument("dir_or_file")
@click.argument("xml_output")
@click.option(
    "--show-steps",
    default=False,
    is_flag=True,
    help="Show steps fo image processing, press any key to get next image/quit.",
)
@click.option(
    "--show-result", default=False, is_flag=True, help="Show result, press any key to quit."
)
def cli(dir_or_file, xml_output, show_steps, show_result):
    """Detect areas of photocoagulation treatment in retina images.

    DIR_OR_FILE - directory with retina images (or single file).

    XML_OUTPUT - Path to XML file.
    """
    src = Source(dir_or_file)
    hull_rec_list = []
    for img_rec in src.images():
        hull_rec_list.append(process(img_rec, show_steps, show_result))
    save_xml(hull_rec_list, xml_output)
