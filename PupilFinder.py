import numpy
import cv2
import os
from VideoTimestampReader import VideoTimestampReader
from scipy.signal import medfilt
from scipy import stats
from skimage.draw import circle_perimeter as DrawCircle
from skimage.io import imsave
from skimage.transform import hough_ellipse as HoughEllipse

def median2way(data, window):
	"""
	Applies median filter forwards and backwards
	@param data:
	@param window:
	@return:
	"""
	return (medfilt(data, window) + medfilt(data[::-1], window)[::-1]) / 2.0


def outliers2nan(data, percentile = 95, absVals = False):
	"""
	Converts outliers beyond a certain  to nan
	@param data: 		1d array
	@param percentile: 	float, percentile threshold
	@param absVals:		bool, absolute values?
	@return:
	"""
	values = data.copy()
	if (absVals):
		values = numpy.abs(values)
	threshold = numpy.percentile(values, percentile)
	mask = values > threshold
	values = data.copy()
	values[mask] = numpy.nan
	return values


class PupilFinder(VideoTimestampReader):
	"""
	Finds the pupil in the videos and generates a trace of it.
	See tweak_eyetrack_preproc
	"""

	def __init__(self, videoFileName = None, window = None, blur = 5, dp = 1, minDistance = 600, param1 = 80,
				 param2 = 20, minRadius = 5, maxRadius = 0, ellipse = False, filter = None, other = None):
		"""
		Constructor
		@param videoFileName:	str?, name of video file to aprse
		@param window: 			4-ple<int>?, subwindow in frame to examine, order [left, right, top, bottom]
		@param blur: 			int, median blur filter width
		@param dp: 				float, inverse ratio of accumulator resolution to image resolution
		@param minDistance: 	float, min distance between centers of detected circles
		@param param1: 			float, higher threshold for canny edge detector
		@param param2: 			float, accumulator threshold at detection stage, smaller => more errors
		@param minRadius: 		int, min circle radius
		@param maxRadius: 		int, max circle radius
		@param ellipse:			bool, use hough ellipse instead of hough circle transform?
		@param filter:			funciton?, filtering function, will override all filters if given
		@param other:			VideoReader?, object to copy contruct from
		"""
		super(PupilFinder, self).__init__(videoFileName, other)
		self.ParseTimestamps()

		self.frames = self.rawFrames		# everything is b/w anyways and we're not modifying the frames

		self.window = window
		self.blur = blur
		# hough transform parameters
		self.ellipse = ellipse
		self.dp = dp
		self.minDistance = minDistance
		self.param1 = param1
		self.param2 = param2
		self.minRadius = minRadius
		self.maxRadius = maxRadius

		self.filter = filter

		# crop to window
		if (self.window is not None):
			self.frames = self.frames[:, self.window[2]:self.window[3], self.window[0]:self.window[1]]

		self.rawPupilLocations = None			# [n x 3] array of x, y, radius
		self.rawGlintLocations = None			# [n x 3] array of x, r, radius of glint location
		self.frameDiffs = None
		self.blinks = None						# [n] array, true when blink is detected
		self.filteredPupilLocations = None
		self.filteredGlintLocations = None


	def FindPupils(self, endFrame = None, bilateral = None, erode = None, filter = None):
		"""
		Find the circles, i.e. pupils in the rawFrames, see eyetrack.video2circles()
		@param endFrame:		int?, frame to read to, defaults to reading all rawFrames
		@param bilateral:		int?, if given, specifies the width of bilateral filter to use
		@param erode:			int?, if not none, erodes and dilates with filter of this size
		@param filter:			function?, filter function to use, if given will oveeride
		@return:
		"""

		if filter is None:
			filter = self.filter

		if ((endFrame is None) or endFrame > self.nFrames):
			endFrame = self.nFrames

		self.frameDiffs = numpy.r_[0, numpy.sum(numpy.diff(self.frames, axis = 0) ** 2, (1, 2))]
		self.blinks = numpy.where(self.frameDiffs > self.frameDiffs.mean() + self.frameDiffs.std() * 2, True, False)

		pupilLocation = []
		### === parallel for ===
		for frame in range(endFrame):
			# eyetrack.find pupil()
			image = self.frames[frame, :, :]
			if filter is not None:
				image = filter(image)
			else:
				if (bilateral is not None):
					image = cv2.bilateralFilter(image, bilateral, 75, 75)
				if (self.blur is not None) and (self.blur > 0):
					image = cv2.medianBlur(image, self.blur)
				if (erode is not None):
					size = 4 * erode
					interval = (2 * erode + 1.) / (size)
					x = numpy.linspace(-erode - interval / 2., erode + interval / 2., size + 1)
					kern1d = numpy.diff(stats.norm.cdf(x))
					kernel_raw = numpy.sqrt(numpy.outer(kern1d, kern1d))
					kernel = kernel_raw / kernel_raw.sum()
					image = cv2.dilate(cv2.erode(image, kernel), kernel)
			circle = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, self.dp, self.minDistance, self.param1, self.param2, self.minRadius, self.maxRadius)
			if (circle is None):
				circle = numpy.zeros(3) * numpy.nan
			circle = circle.squeeze()
			if (self.window is not None):
				circle[0] += self.window[0]
				circle[1] += self.window[2]

			pupilLocation.append(circle)
		self.rawPupilLocations = numpy.asarray(pupilLocation)


	def FilterPupils(self, windowSize = 15, outlierThresholds = None, filterPupilSize = True):
		"""
		Filters raw pupil locations
		@param windowSize:			int, median filter time window size
		@param outlierThresholds:	list<float>?, thresholds in percentiles at which to nan outliers, if none, does not nan outliers
		@param filterPupilSize:		bool, filter pupil size alone with position?
		@return:
		"""
		if (self.rawPupilLocations is None):
			self.FindPupils()

		self.filteredPupilLocations = self.rawPupilLocations.copy()
		for i in range(3 if filterPupilSize else 2):
			self.filteredPupilLocations[:, i] = median2way(self.filteredPupilLocations[:, i], windowSize)

		for i in range(self.nFrames):
			if self.blinks[i]:
				self.filteredPupilLocations[i, :] = numpy.nan

		if (outlierThresholds is not None):
			for i in range(3):
				if (outlierThresholds[i] is not None):
					self.filteredPupilLocations[:, i] = outliers2nan(self.filteredPupilLocations[:, i], outlierThresholds[i])
			# 1 nan in row => entire row nan
			self.filteredPupilLocations[numpy.isnan(self.filteredPupilLocations.sum(axis = -1))] = numpy.nan


	def WritePupilFrames(self, directory, startFrame = None, endFrame = None, filtered = True, filteredFrames = False):
		"""
		Draws frames back out with the pupil circled
		@param directory: 		str, directory to which to save
		@param startFrame:		int?, first frame to draw
		@param endFrame: 		int?, last frame to draw, defaults to all of them
		@param filtered:		bool, use filtered trace instead of unfiltered?
		@param filteredFrames:	bool, use filtered frames instead of raw frames?
		@return:
		"""
		if (startFrame is None):
			startFrame = 0
		if (endFrame is None):
			endFrame = self.nFrames

		if (self.rawPupilLocations is None):
			self.FindPupils()

		if (filtered and (self.filteredPupilLocations is None)):
			self.FilterPupils()

		circles = self.filteredPupilLocations.astype(numpy.int) if filtered else self.rawPupilLocations.astype(numpy.int)

		if not os.path.exists(directory):
			os.makedirs(directory)

		image = numpy.zeros([self.height, self.width, 3])
		### === parallel for ===
		for frame in range(startFrame, endFrame):
			for i in range(3):
				if filteredFrames:
					if self.window is None:
						image[:, :, i] = self.frames[frame, :, :]
					else:
						image[:, :, i] = self.rawFrames[frame, :, :]
						image[self.window[2]:self.window[3], self.window[0]:self.window[1], i] = self.frames[frame, :, :]
				else:
					image[:, :, i] = self.rawFrames[frame, :, :]
			if not (filtered and self.filteredPupilLocations[frame, 0] == numpy.nan):
				if (not self.blinks[frame]) and (not numpy.any(numpy.isnan(self.filteredPupilLocations[frame, :]))):
					for radiusOffset in range(-2, 3):
						y, x = DrawCircle(circles[frame, 0], circles[frame, 1], circles[frame, 2] + radiusOffset, shape = (self.height, self.width))
						image[x, y, 2] = 255
						image[(circles[frame, 1] - 4):(circles[frame, 1] + 4), (circles[frame, 0] - 1):(circles[frame, 0] + 1), 2] = 255
						image[(circles[frame, 1] - 1):(circles[frame, 1] + 1), (circles[frame, 0] - 4):(circles[frame, 0] + 4), 2] = 255
			imsave(directory + '/frame_{:06d}.png'.format(frame), image[:, :, ::-1])


	def WritePupilVideo(self, fileName, startFrame = None, endFrame = None, filtered = True, filteredFrames = False):
		"""
		Writes a video instead
		@param fileName:
		@param startFrame:		int?, first frame to draw
		@param endFrame: 		int?, last frame to draw, defaults to all of them
		@param filtered:		bool, use filtered trace instead of unfiltered?
		@param filteredFrames:	bool, used filtered instead of raw frames?
		@return:
		"""
		if (startFrame is None):
			startFrame = 0
		if (endFrame is None):
			endFrame = self.nFrames

		if (self.rawPupilLocations is None):
			self.FindPupils()

		if (filtered and (self.filteredPupilLocations is None)):
			self.FilterPupils()

		circles = self.filteredPupilLocations.astype(numpy.int) if filtered else self.rawPupilLocations.astype(numpy.int)

		video = cv2.VideoWriter(fileName, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), self.fps, (self.width, self.height))
		image = numpy.zeros([self.height, self.width, 3], dtype = numpy.uint8)
		for frame in range(startFrame, endFrame):
			for i in range(3):
				if filteredFrames:
					if self.window is None:
						image[:, :, i] = self.frames[frame, :, :]
					else:
						image[:, :, i] = self.rawFrames[frame, :, :]
						image[self.window[2]:self.window[3], self.window[0]:self.window[1], i] = self.frames[frame, :, :]
				else:
					image[:, :, i] = self.rawFrames[frame, :, :]
			if not (filtered and self.filteredPupilLocations[frame, 0] == numpy.nan):
				if (not self.blinks[frame]) and (not numpy.any(numpy.isnan(self.filteredPupilLocations[frame, :]))):
					for radiusOffset in range(-2, 3):
						y, x = DrawCircle(circles[frame, 0], circles[frame, 1], circles[frame, 2] + radiusOffset, shape = (self.height, self.width))
						image[x, y, 2] = 255
						image[(circles[frame, 1] - 4):(circles[frame, 1] + 4), (circles[frame, 0] - 1):(circles[frame, 0] + 1), 2] = 255
						image[(circles[frame, 1] - 1):(circles[frame, 1] + 1), (circles[frame, 0] - 4):(circles[frame, 0] + 4), 2] = 255
			video.write(image)
		video.release()