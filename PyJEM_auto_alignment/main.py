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
from time import sleep
from functools import partial


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
        image_array = np.frombuffer(bufferimg, dtype=np.uint8)
        img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        # img = Image.open(io.BytesIO(image_array))
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
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    # plt.imshow(gray_image, cmap='gray')
    # plt.show()
    # Check that there is beam.
    image_8bit = cv2.normalize(gray_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    if not any(np.array(image_8bit).flatten() > beamthreshold):
        raise NoBeamError()

    _, thresholded_img = cv2.threshold(gray_image, 0, contourthreshold, cv2.THRESH_BINARY)
    contour, _ = cv2.findContours(thresholded_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

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

    # Stop if contours are touching edges. #TODO test case on scope doesnt throw error.
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


class HTOffError(ValueError):
    pass


class EmissionOffError(ValueError):
    pass


class BeamValveClosedError(ValueError):
    pass


class BeamBlankedError(ValueError):
    pass


class HolderNotInserted(ValueError):
    pass


class NotMagMode(ValueError):
    pass


def confirm_microscope_state(offline=False):
    if offline:
        return True
    # Check HT on.
    if TEM3.HT3().GetHtValue() != 200000.0:
        raise HTOffError()
    # Check emission on.
    if TEM3.GUN3().GetEmissionCurrentValue() == 0.0:
        raise EmissionOffError()
    # Check beam valve open.
    if TEM3.FEG3().GetBeamValve() == 0:  # 0=closed
        raise BeamValveClosedError()
    # Beam not blanked.
    if TEM3.Def3().GetBeamBlank() == 1:  # 1=ON
        raise BeamBlankedError()
    # Holder is inserted.
    if TEM3.Stage3().GetHolderStts() == 0:  # 0=OUT
        raise HolderNotInserted()
    # TODO get TEM mode and illumination mode
    # Check MAG mode.
    if TEM3.EOS3().GetFunctionMode()[0] == 0:  # TEM MAG mode
        raise NotMagMode()
    # Start camera view and set exposure/gain settings.
    detector.Detector('TVCAM_SCR_L').livestart()
    detector.Detector('TVCAM_SCR_L').set_exposuretime_value(120)  # 0.12 s
    detector.Detector('TVCAM_SCR_L').set_gainindex(0.5)  # actually sets gain to 1

    return True


def beamShift(delta_x, delta_y, shiftscale):
    ix, iy = TEM3.Def3().GetCLA1()
    TEM3.Def3().SetCLA1(ix + delta_x * shiftscale, iy - delta_y * shiftscale)


def apertureShift(delta_x, delta_y, KIND, SIZE, shiftscale):
    # left x: lowers ax value, moves beam up.
    # up y:  increases ay value, moves beam right.
    # Initialise aperture.
    TEM3.Apt3().SelectKind(KIND)
    TEM3.Apt3().SetSize(SIZE)
    ix, iy = TEM3.Apt3().GetPosition()
    TEM3.Apt3().SetPosition(ix + delta_x * shiftscale, iy - delta_y * shiftscale)


def centerBeam(shift_function, tolerance=10, max_iterations=10):
    """
    Iteratively shifts beam until it is centered within tolerance.
    """
    for iteration in range(max_iterations):
        try:
            minbeamsize = 2
            image, contour = getBeamContour(show=False, beamthreshold=50, contourthreshold=190, minbeamsize=minbeamsize,
                                            contoursmax=1, getarea=False)

            height, _, _ = image.shape  # assume height = width

            # Get contour center
            M = cv2.moments(contour[0])
            if M['m00'] == 0:
                ValueError("Zero contour area found, there is an issue.")
                continue

            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])

            # Check if centered
            if abs(cx - height // 2) <= tolerance and abs(cy - height // 2) <= tolerance:
                print(f"Beam is centered (iteration {iteration + 1}).")
                break

            # Calculate shift
            delta_cx = cx - height // 2
            delta_cy = cy - height // 2
            print(f"shift x {delta_cx} : shift y {delta_cy}")

            try:
                shift_function(delta_cx, delta_cy)
                print(f"Iteration {iteration + 1}: Shifted by ({delta_cx}, {delta_cy})")
            except:
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
    confirm_microscope_state()
    centerBeam(shift_function=partial(apertureShift, KIND=1, SIZE=2, shiftscale=1))
    centerBeam(shift_function=partial(beamShift, shiftscale=1))

    # print(collectMetadata())

    # TEM3.Apt3().SelectKind(1)
    # TEM3.Apt3().SetSize(2)

    # ax,ay = TEM3.Apt3().GetPosition()
    # print(ax,ay)
    # left x: lowers ax value, moves beam up.
    # up y:  increases ay value, moves beam right.
    # TEM3.Apt3().SetPosition(ax-20,ay)

    # image = collectPhosphorImage()
    # beamsize = getBeamSize()
    # meta = collectMetadata()
