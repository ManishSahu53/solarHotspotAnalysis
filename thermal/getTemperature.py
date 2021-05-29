from thermal import getThermal
import numpy as np


fir = getThermal.FlirImageExtractor(exiftool_path='thermal/Image-ExifTool-11.85/exiftool')

def getTemp(path_image):
    fir.process_image(path_image)
    return fir.extract_thermal_image()


def getStas(tempArray, bbox):
    """
    To Calculate temperature statistics of bbox
    bbox = [x1, y1, x2, y2]
    [col1, row1, col2, row2]
    """
    stats = {
        'min': 0,
        'max': 0,
        'mean': 0,
        'median': 0,
        'area(inPixels)': 0
    }

    hissa = tempArray[bbox[1]: bbox[3], bbox[0]: bbox[2]]
    stats['min'] = np.min(hissa)
    stats['max'] = np.max(hissa)
    stats['mean'] = np.mean(hissa)
    stats['median'] = np.median(hissa)
    stats['area(inPixels)'] = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    return stats
