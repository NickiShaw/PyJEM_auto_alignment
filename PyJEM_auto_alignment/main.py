try:
    from PyJEM import TEM3, detector
    offline = False
except:
    from PyJEM.offline import TEM3, detector
    offline = True

import numpy as np
import matplotlib.pyplot as plt
import cv2
import math as m
from scipy.optimize import curve_fit
from statistics import mean
from matplotlib import pyplot as plt
from scipy.special import expit, logit
from PIL import Image
import io


class TEMControl:
    def __init__(self):
        self.stage_object = TEM3.Stage3()

    '''
    self.stage_object.SetX(val)
    self.stage_object.SetY(val)
    self.stage_object.SetZ(val)
    self.stage_object.SetTiltXAngle(val)
    self.stage_object.SetTiltYAngle(val)
    '''


def collectPhosphorImage(show=False):
    if offline:
        image_dir = "test.tif"
        img = cv2.imread(image_dir, 1)

    else:
        bufferimg = detector.Detector('TVCAM_SCR_L').snapshot('tiff')
        img = Image.open(io.BytesIO(bufferimg))
    if show:
        plt.imshow(img, cmap='gray')
        plt.show()
    return img


def collectMetadata():
    mag = TEM3.EOS3().GetMagValue()[0]
    spot = TEM3.EOS3().GetSpotSize() + 1
    angle = TEM3.EOS3().GetAlpha() + 1
    brightness = TEM3.Lens3().GetCL3()

    return {'MAG': mag, 'SPOT': spot, 'ANGLE': angle, 'BRI': brightness}

class NoBeamError(ValueError):
    pass

class TooManyContoursError(ValueError):
    pass

class BeamOnEdgeError(ValueError):
    pass

class TooSmallContoursError(ValueError):
    pass

def getBeamContour(show=False, beamthreshold=50, contourthreshold=190, minbeamsize=2, contoursmax=1, getarea=False):
    image = collectPhosphorImage()
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Check that there is beam.
    image_8bit = cv2.normalize(gray_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    if not any(np.array(image_8bit).flatten() > beamthreshold):
        raise NoBeamError()

    _, thresholded_img = cv2.threshold(gray_image, 0, contourthreshold, cv2.THRESH_BINARY)
    contour, _ = cv2.findContours(thresholded_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # plt.imshow(thresholded_img, cmap='gray')
    # plt.show()

    filtered_contours = []
    for c in contour:
        # remove small contours
        if cv2.contourArea(c) > minbeamsize:
            filtered_contours.append(c)

    # Stop if there are no contours left.
    if len(filtered_contours) == 0:
        raise TooSmallContoursError()

    # Stop if there are too many contours.
    if len(filtered_contours) > contoursmax:
        raise TooManyContoursError()

    # Stop if contours are touching edges.
    height, width, channels = image.shape
    for i, c in enumerate(filtered_contours):
        flat_contours = np.array(c).flatten()
        if any(flat_contours == int(height)) or any(flat_contours == int(width)):
            raise BeamOnEdgeError()
    if show:
        for c in filtered_contours:
            cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
        plt.imshow(image)
        plt.show()

    if getarea:
        # Draw contours and return area.
        areas = []
        for c in filtered_contours:
            area = cv2.contourArea(c)
            areas.append(area)
            # print(f"Area of the contour: {area} pixels")
        return areas
    else:
        return image, filtered_contours


def confirm_tilt_state(offline=False):
    '''
    Confirms the tilt state of the TEM.
    :return: True if the TEM is in tilt state, False otherwise.
    '''
    if offline:
        return True

    # HT and emission on.
    if TEM3.HT3().GetHtValue() != 200000.0:
        return False
    # TODO look for GetEmissionOnStatus
    # Beam valve open.
    if TEM3.FEG3().GetBeamValve() == 0:
        return False
    # Beam not blanked.
    if TEM3.Def3().GetBeamBlank() == 1: # 1=ON
        return False
    # Holder is inserted.
    if TEM3.Stage3().GetHolderStts() == 0: # 0=OUT
        return False
    # CL 2 inserted, no other apertures are inserted.
    # 1=CLA, 2=OLA, 3=HCA, 4=SAA, 5=ENTA (N/A), 6=EDS
    for apt in [1,2,3,4,6]:
        TEM3.Apt3().SelectKind(apt)
        if apt == 1:
            TEM3.Apt3().SelectSize(2)
        else:
            TEM3.Apt3().SelectSize(0)
    # Focusing screen not inserted.
    # TEM mode, TEM illumination.
    if TEM3.EOS3.GetFunctionMode()[0] == 0: # TEM MAG mode
        return False
    # Mag should be 100kx.
    if TEM3.EOS3().GetMagValue() != 100000:
        return False
    # TODO set brightness value

    detector.Detector(0).livestart()
    detector.Detector(0).set_exposuretime_value(120)  # 0.12 s
    detector.Detector(0).set_gainindex(0.5)  # actually sets gain to 1

    return True

def centerBeam(shiftscale=1, tolerance=40, shift_function="CLA1", max_iterations=10):
    """
    Iteratively shifts beam until it is centered within tolerance.
    """
    for iteration in range(max_iterations):
        try:
            minbeamsize = 2
            image, contour = getBeamContour(show=False, beamthreshold=50, contourthreshold=190, minbeamsize=minbeamsize, contoursmax=1, getarea=False)

            height, _, _ = image.shape  # assume height = width

            # Get contour center
            M = cv2.moments(contour[0])
            if M['m00'] == 0:
                ValueError("Zero contour area found, there is an issue.")
                continue

            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])

            # Check if centered
            if abs(cx - height//2) <= tolerance and abs(cy - height//2) <= tolerance:
                print(f"Beam is centered (iteration {iteration+1}).")
                break

            # Calculate shift
            delta_cx = cx - height
            delta_cy = cy - height

            if shift_function == "CLA1":
                ix, iy = TEM3.Def3().GetCLA1()
                TEM3.Def3().SetGunA1(ix + delta_cx * shiftscale,
                                     iy + delta_cy * shiftscale)
                print(f"Iteration {iteration+1}: Shifted by ({delta_cx}, {delta_cy})")
            else:
                raise ValueError("alignBeam did not receive a valid 'shift_function'")

        except NoBeamError:
            print("No beam detected — adjust alignment.")
            break
        except TooManyContoursError:
            print("Too many contours — lower contour threshold or clean image.")
            break
        except TooSmallContoursError:
            print(f"Contours were all removed by filtering, minbeamsize was {minbeamsize}")
            break
        except BeamOnEdgeError:
            print("Beam is on edge — recenter.")
            break
    else:
        print("Max iterations reached, beam may not be centered.")




if __name__ == "__main__":
    centerBeam()
    # image = collectPhosphorImage()
    # beamsize = getBeamSize()
    # meta = collectMetadata()