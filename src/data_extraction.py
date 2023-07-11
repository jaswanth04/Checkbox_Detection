import cv2
from abc import ABC, abstractmethod
import json
import numpy as np

AREA_THRESHOLD = 40


class TRFDataExtractor(ABC):
    def __init__(self, roi_data) -> None:
        super().__init__()
        with open(roi_data, 'rb') as roi_data_file:
            roi = json.load(roi_data_file)
            shapes = []
            regions = []
            for item in roi['data']['regions']:
                shapes.append(item['shape_attributes'])
                regions.append(item['region_attributes'])

            self._roi = list(zip(shapes, regions))

    def _get_edges(self, im: cv2.Mat) -> cv2.Mat:
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(
            blur, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        edges = cv2.Canny(blur, 220, 250)

        return edges

    @abstractmethod
    def extract(self, aligned_image: cv2.Mat) -> list:
        pass


class BoxDataExtractor(TRFDataExtractor):
    def __init__(self, roi_data, non_black_count_threshold=90) -> None:
        super().__init__(roi_data)
        self._non_black_count_threshold = non_black_count_threshold

    def extract(self, aligned_image: cv2.Mat) -> list:
        entity_list = []

        edges = self._get_edges(aligned_image)

        for shape, region in self._roi:
            x = shape['x']
            y = shape['y']
            w = shape['width']
            h = shape['height']

            im = edges[y:y+h, x:x+w]
            non_black_count = cv2.countNonZero(im)
            # print(
            #     f'Region: {region["type"]}, Non Black Count: {non_black_count}')
            if non_black_count > self._non_black_count_threshold:
                entity_list.append(region['type'])

        return entity_list


class MorphDataExtractor(TRFDataExtractor):
    def __init__(self, roi_data, non_black_count_threshold=0) -> None:
        super().__init__(roi_data)
        self._non_black_count_threshold = non_black_count_threshold

    def _clean_image(self, im: cv2.Mat) -> cv2.Mat:
        cnts, _ = cv2.findContours(
            im, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]

        for c in cnts:
            area = cv2.contourArea(c)
            if area < AREA_THRESHOLD:
                cv2.drawContours(im, [c], -1, 0, -1)

        return im

    def extract(self, aligned_image: cv2.Mat) -> list:
        entity_list = []
        gray = cv2.cvtColor(aligned_image, cv2.COLOR_BGR2GRAY)

        for shape, region in self._roi:

            # Extracting the roi
            x = shape['x']
            y = shape['y']
            w = shape['width']
            h = shape['height']

            # Extracting the ROI from Image
            im = gray[y:y+h, x:x+w]

            # Binarization of the Image
            thresh = cv2.adaptiveThreshold(
                im, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 10)

            # Cleaning small noises using contours
            thresh = self._clean_image(thresh)

            # Creating the horizontal kernel and detecting horizontal lines
            repair_kernel_horizontal = cv2.getStructuringElement(
                cv2.MORPH_RECT, (4, 1))
            repair_horizontal = cv2.morphologyEx(
                thresh, cv2.MORPH_OPEN, repair_kernel_horizontal, iterations=2)

            # Detecting horizontal lines using Above Morph from the above morph and  and drawing them on the image
            horizontal_lines = cv2.HoughLinesP(
                repair_horizontal, 1, np.pi/180, 20, 10, 10)

            if horizontal_lines is not None:
                for line in horizontal_lines:
                    for x1, y1, x2, y2 in line:
                        theta = np.arctan(
                            (y2 - y1) / (x2 - x1 + 1e-8)) * 180 / np.pi
                        if (theta > -6 and theta < 6):
                            cv2.line(thresh, (x1, y1), (x2, y2), (0, 0, 0), 2)

            # Creating the vertical kernel and detecting vertical lines
            repair_kernel_vertical = cv2.getStructuringElement(
                cv2.MORPH_RECT, (1, 4))
            repair_vertical = cv2.morphologyEx(
                thresh, cv2.MORPH_OPEN, repair_kernel_vertical, iterations=1)

            # Detecting Vertical Lines using Above vertical morph using hough lines and drawing them on the thresholded image
            vertical_lines = cv2.HoughLinesP(
                repair_vertical, 1, np.pi/180, 12, 40, 4)
            if vertical_lines is not None:
                for line in vertical_lines:
                    for x1, y1, x2, y2 in line:
                        theta = np.abs(
                            np.arctan((y2 - y1)/(x2 - x1 + 1e-8)) * 180 / np.pi)
                        if (theta > 84 and theta < 96):
                            cv2.line(thresh, (x1, y1), (x2, y2), (0, 0, 0), 2)

            # Final Cleaning of small parts in the image
            thresh = self._clean_image(thresh)

            non_black_count = cv2.countNonZero(thresh)

            if non_black_count > self._non_black_count_threshold:
                entity_list.append(region['type'])

        return entity_list


# class MorphDataExtractorWithFallback()
