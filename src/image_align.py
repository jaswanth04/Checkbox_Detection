from abc import ABC, abstractmethod
import cv2
import numpy as np


class ImageAligner(ABC):
    def __init__(self, template: str) -> None:
        super().__init__()
        template_im: cv2.Mat = cv2.imread(template)
        self._template: cv2.Mat = cv2.cvtColor(template_im, cv2.COLOR_BGR2GRAY)

    @abstractmethod
    def align(self, query_image: cv2.Mat) -> cv2.Mat:
        pass


class SIFTAligner(ImageAligner):
    def __init__(self, template: str) -> None:
        super().__init__(template)
        self._h, self._w = self._template.shape
        self._sift = cv2.SIFT_create()
        self._kps_template, self._desc_template = self._sift.detectAndCompute(
            self._template, None)
        self._matcher = cv2.BFMatcher()

    # def align(self, query_image: str) -> cv2.Mat:
    #     query = cv2.imread(query_image)
    #     self.align(query)

    def align(self, query_image: cv2.Mat) -> cv2.Mat:
        # query_image = cv2.imread(query_image_path)
        query = cv2.cvtColor(query_image, cv2.COLOR_BGR2GRAY)
        kps_query, desc_query = self._sift.detectAndCompute(query, None)

        matches = self._matcher.knnMatch(desc_query, self._desc_template, k=2)
        # Apply ratio test
        good = []
        for m, n in matches:
            if m.distance < 0.75*n.distance:
                good.append([m])

        pts_template = np.zeros((len(good), 2), dtype="float")
        pts_query = np.zeros((len(good), 2), dtype="float")

        for i, m in enumerate(good):
            try:
                pts_query[i, :] = kps_query[m[0].queryIdx].pt
                pts_template[i, :] = self._kps_template[m[0].trainIdx].pt
            except IndexError:
                print(f'Index Error at {i}')

        H, mask = cv2.findHomography(
            pts_query, pts_template, method=cv2.RANSAC)
        aligned = cv2.warpPerspective(query_image, H, (self._w, self._h))

        return aligned


class ORBAligner(ImageAligner):
    def __init__(self, template: str, max_features=3000, keep_percent=0.2) -> None:
        super().__init__(template)
        self._h, self._w = self._template.shape
        self._orb = cv2.ORB_create(max_features)

        self._kps_template, self._desc_template = self._orb.detectAndCompute(
            self._template, None)
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self._keep_percent = keep_percent

    def align(self, query_image: cv2.Mat) -> cv2.Mat:
        query = cv2.cvtColor(query_image, cv2.COLOR_BGR2GRAY)
        kps_query, desc_query = self._orb.detectAndCompute(query, None)

        matches = self._matcher.match(desc_query, self._desc_template)

        matches = sorted(matches, key=lambda x: x.distance, reverse=False)

        keep = int(len(matches) * self._keep_percent)
        matches = matches[:keep]

        pts_template = np.zeros((len(matches), 2), dtype="float")
        pts_query = np.zeros((len(matches), 2), dtype="float")

        for i, m in enumerate(matches):
            pts_query[i, :] = kps_query[m.queryIdx].pt
            pts_template[i, :] = self._kps_template[m.trainIdx].pt

        H, mask = cv2.findHomography(
            pts_query, pts_template, method=cv2.RANSAC)
        aligned_orb = cv2.warpPerspective(query_image, H, (self._w, self._h))

        return aligned_orb
