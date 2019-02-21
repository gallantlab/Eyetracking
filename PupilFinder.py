import numpy
import cv2
import os
from VideoTimestampReader import VideoTimestampReader
from scipy.signal import medfilt
from skimage.draw import circle_perimeter as DrawCircle
from skimage.io import imsave

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
				 param2 = 20, minRadius = 5, maxRadius = 0, other = None):
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
		@param other:			VideoReader?, object to copy contruct from
		"""
		super(PupilFinder, self).__init__(videoFileName, other)
		# self.frames = self.rawFrames.mean(-1).astype(numpy.uint8)	# average over the color dimensions

		self.window = window
		self.blur = blur
		# hough transform parameters
		self.dp = dp
		self.minDistance = minDistance
		self.param1 = param1
		self.param2 = param2
		self.minRadius = minRadius
		self.maxRadius = maxRadius

		# crop to window
		# if (self.window is not None):
		# 	self.frames = self.frames[:, self.window[2]:self.window[3], self.window[0]:self.window[1]]

		self.rawPupilLocations = None			# [n x 3] array of x, y, radius
		self.frameDiffs = None
		self.blinks = None						# [n] array, true when blink is detected
		self.filteredPupilLocations = None


	def FindPupils(self, endFrame = None, bilateral = None):
		"""
		Find the circles, i.e. pupils in the rawFrames, see eyetrack.video2circles()
		@param endFrame:		int?, frame to read to, defaults to reading all rawFrames
		@param bilateral:		int?, if present, radius to use for surface blur
		@return:
		"""
		if ((endFrame is None) or endFrame > self.nFrames):
			endFrame = self.nFrames

		self.frameDiffs = numpy.r_[0, numpy.sum(numpy.diff(self.rawFrames, axis = 0) ** 2, (1, 2, 3))]
		self.blinks = numpy.where(self.frameDiffs > self.frameDiffs.mean() + self.frameDiffs.std() * 2, True, False)

		pupilLocation = []
		### === parallel for ===
		for frameIndex in range(endFrame):
			# eyetrack.find pupil()
			self.frame = self.rawFrames[frameIndex, self.window[2]:self.window[3], self.window[0]:self.window[1], :].mean(-1).astype(numpy.uint8)
			if (bilateral is not None) and (bilateral > 0):
				self.frame = cv2.bilateralFilter(self.frame, bilateral, 100, 75)
				self.frame = cv2.medianBlur(self.frame, self.blur)
			else:
				self.frame = cv2.medianBlur(self.frames[frameIndex, :, :], self.blur)
			circle = cv2.HoughCircles(self.frame, cv2.HOUGH_GRADIENT, self.dp, self.minDistance, self.param1, self.param2, self.minRadius, self.maxRadius)
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


	def WritePupilFrames(self, directory, startFrame = None, endFrame = None, filtered = True, burnLocation = True):
		"""
		Draws frames back out with the pupil circled
		@param directory: 		str, directory to which to save
		@param startFrame:		int?, first frame to draw
		@param endFrame: 		int?, last frame to draw, defaults to all of them
		@param filtered:		bool, use filtered trace instead of unfiltered?
		@param burnLocation:	bool, burn location of pupil into image?
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

		### === parallel for ===
		for frame in range(startFrame, endFrame):
			image = self.rawFrames[frame, :, :, :].copy()
			# image[:, :, 1] = image[:, :, 0]
			# image[:, :, 2] = image[:, :, 0]
			if not (filtered and self.filteredPupilLocations[frame, 0] == numpy.nan):
				if (not self.blinks[frame]) and (not numpy.any(numpy.isnan(self.filteredPupilLocations[frame, :]))):
					for radiusOffset in range(-2, 3):
						y, x = DrawCircle(circles[frame, 0], circles[frame, 1], circles[frame, 2] + radiusOffset, shape = (self.height, self.width))
						image[x, y, 2] = 255
						image[(circles[frame, 1] - 4):(circles[frame, 1] + 4), (circles[frame, 0] - 1):(circles[frame, 0] + 1), 2] = 255
						image[(circles[frame, 1] - 1):(circles[frame, 1] + 1), (circles[frame, 0] - 4):(circles[frame, 0] + 4), 2] = 255
					if burnLocation:
						cv2.putText(image, 'x: {:03d} y: {:03d} r: {:03d}'.format(circles[frame, 0], circles[frame, 1], circles[frame, 2]), (30, 20), cv2.FONT_HERSHEY_DUPLEX, 0.75, [0, 255, 0])
						cv2.putText(image, 'frame {:06d}'.format(frame), (60, 20), cv2.FONT_HERSHEY_DUPLEX, 0.75, [0, 255, 0])
			imsave(directory + '/frame_{:06d}.png'.format(frame), image[:, :, ::-1])


	def WritePupilVideo(self, fileName, startFrame = None, endFrame = None, filtered = True, burnLocation = True):
		"""
		Writes a video with the pupil circled
		@param fileName:
		@param startFrame:		int?, first frame to draw
		@param endFrame: 		int?, last frame to draw, defaults to all of them
		@param filtered:		bool, use filtered trace instead of unfiltered?
		@param burnLocation:	bool, burn location of pupil into image?
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
		image = numpy.zeros_like(self.rawFrames[0, :, :, :])
		for frame in range(startFrame, endFrame):
			image = self.rawFrames[frame, :, :, :].copy()
			# image[:, :, 1] = image[:, :, 0]
			# image[:, :, 2] = image[:, :, 0]
			if not (filtered and self.filteredPupilLocations[frame, 0] == numpy.nan):
				if (not self.blinks[frame]) and (not numpy.any(numpy.isnan(self.filteredPupilLocations[frame, :]))):
					for radiusOffset in range(-2, 3):
						y, x = DrawCircle(circles[frame, 0], circles[frame, 1], circles[frame, 2] + radiusOffset, shape = (self.height, self.width))
						image[x, y, 2] = 255
						image[(circles[frame, 1] - 4):(circles[frame, 1] + 4), (circles[frame, 0] - 1):(circles[frame, 0] + 1), 2] = 255
						image[(circles[frame, 1] - 1):(circles[frame, 1] + 1), (circles[frame, 0] - 4):(circles[frame, 0] + 4), 2] = 255
					if burnLocation:
						cv2.putText(image, 'x: {:03d} y: {:03d} r: {:03d}'.format(circles[frame, 0], circles[frame, 1], circles[frame, 2]), (10, 20), cv2.FONT_HERSHEY_DUPLEX, 0.75, [0, 255, 0])
				else:
					if burnLocation:
						cv2.putText(image, 'Blink', (10, 20), cv2.FONT_HERSHEY_DUPLEX, 0.75, [0, 255, 0])
				if burnLocation:
					cv2.putText(image, 'frame {:06d}'.format(frame), (10, 45), cv2.FONT_HERSHEY_DUPLEX, 0.75, [0, 255, 0])
			video.write(image)
		video.release()


	def GetTraces(self, filtered = True, fps = None):
		"""
		Resamples traces to a different fps using closest frame
		@param filtered:
		@param fps:			int?, if present, resample the traces
		@return:
		"""
		traces = self.filteredPupilLocations if filtered else self.rawPupilLocations
		if fps is None:
			return traces

		if (fps - self.fps) < 0.1:
			return traces
		# outTrace = numpy.zeros([int(self.duration * fps), 4])

		# dt = 1000.0 / float(fps)	# in ms
		# hour = self.time[0, 0]
		# minute = self.time[0, 1]
		# second = self.time[0, 2]
		# millisecond = self.time[0, 3]
		#
		# for frame in range(outTrace.shape[0]):
		# 	time = int(dt * frame)
		# 	dH = time / (60 * 60 * 1000)
		# 	time -= dH * (60 * 60 * 1000)
		# 	dM = time / (60 * 1000)
		# 	time -= dM * (60 * 1000)
		# 	dS = time / (1000)
		# 	dMS = time % 1000
		#
		# 	closestFrame = self.FindOnsetFrame(hour + dH, minute + dM, second + dS, millisecond + dMS)
		# 	outTrace[frame, :] = traces[closestFrame, :]

		outTrace = []
		for i in range(0, traces.shape[0], 2):
			outTrace.append(traces[i, :])
		outTrace = numpy.array(outTrace)

		return outTrace