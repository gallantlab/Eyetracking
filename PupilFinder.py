import numpy
import cv2
import os
import threading
from zipfile import ZipFile
from .EyetrackingUtilities import ReadNPY, SaveNPY
from .VideoTimestampReader import VideoTimestampReader
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
	Uses the hough transform.
	See tweak_eyetrack_preproc
	"""

	@staticmethod
	def HoughWorker(rawFrames, window, bilateral, blur, dp, minDistance, param1, param2,
					minRadius, maxRadius, rawPupilLocations):
		"""
		For multithreading
		@param rawFrames:
		@param radii:
		@param window:
		@param bilateral:
		@param blur:
		@return:
		"""
		pupilLocation = []
		### === parallel for ===
		for frameIndex in range(rawFrames.shape[0]):
			# eyetrack.find pupil()
			frame = rawFrames[frameIndex, window[2]:window[3], window[0]:window[1], :].mean(-1).astype(numpy.uint8)
			if (bilateral is not None) and (bilateral > 0):
				frame = cv2.bilateralFilter(frame, bilateral, 100, 75)
				frame = cv2.medianBlur(frame, blur)
			else:
				frame = cv2.medianBlur(rawFrames[frameIndex, :, :].mean(-1).astype(numpy.uint8), blur)
			circle = cv2.HoughCircles(frame, cv2.HOUGH_GRADIENT, dp, minDistance, param1, param2, minRadius, maxRadius)
			if (circle is None):
				circle = numpy.zeros(3) * numpy.nan
			circle = circle.squeeze()
			if (window is not None):
				circle[0] += window[0]
				circle[1] += window[2]

			rawPupilLocations[frameIndex, :] = circle


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
		"""
		@ivar: Window in frame to look for the pupil in [left, right, top bottom]
		@type: tuple<int, int, int, int>
		"""
		self.blur = blur
		"""
		@ivar: Median filter width
		@type: int
		"""
		# hough transform parameters
		self.dp = dp
		"""
		@ivar: Inverse ration of accumulator resolution to image resolution in Hough transform
		@type: float
		"""
		self.minDistance = minDistance
		"""
		@ivar: Minimum distance in pixels between multiple detected circules by Hough transform
		@type: float
		"""
		self.param1 = param1
		"""
		@ivar: upper threshold for Canny edge detector in Hough transform
		@type: float
		"""
		self.param2 = param2
		"""
		@ivar: accumulator threashold for Hough transform. Smaller => more errors
		@type: float
		"""
		self._minRadius = minRadius
		"""
		@ivar: Minimum circle radius in pixels
		@type: int
		"""
		self._maxRadius = maxRadius
		"""
		@ivar: Maximum circle radius in pixels
		@type: int
		"""

		# crop to window
		# if (self.window is not None):
		# 	self.frames = self.frames[:, self.window[2]:self.window[3], self.window[0]:self.window[1]]

		self.rawPupilLocations = None			# [n x 3] array of x, y, radius
		"""
		@ivar: Raw pupil locations read from video file, columns are [x, y, radius]
		@type: numpy.ndarray
		"""
		self.frameDiffs = None
		"""
		@ivar: Diffs between each successive frame for pupil locations
		@type:	 numpy.ndarray
		"""
		self.blinks = None						# [n] array, true when blink is detected
		"""
		@ivar: Is there a blink this frame?
		@type: list<bool>
		"""
		self.filteredPupilLocations = None
		"""
		@ivar: Pupil locations that have been temporally filtered
		@type: numpy.ndarray
		"""


	@property
	def minRadius(self):
		"""
		Minimum possible radius of pupil
		@return:
		@rtype: int
		"""
		return self._minRadius


	@minRadius.setter
	def minRadius(self, value):
		self._minRadius = value


	@property
	def maxRadius(self):
		"""
		Max possible radius of pupil
		@return:
		@rtype: int
		"""
		return self._maxRadius


	@maxRadius.setter
	def maxRadius(self, value):
		self._maxRadius = value


	def InitFromOther(self, other):
		"""
		Jank copy constructor
		@param other: 	other object
		@type other:	PupilFinder
		"""
		super(PupilFinder, self).InitFromOther(other)
		self.window = other.window
		self.blur = other.blur
		self.dp = other.dp
		self.minDistance = other.minDistance
		self.param1 = other.param1
		self.param2 = other.param2
		self._minRadius = other._minRadius
		self._maxRadius = other._maxRadius
		if other.rawPupilLocations is not None:
			self.rawPupilLocations = other.rawPupilLocations.copy()
		if other.frameDiffs is not None:
			self.frameDiffs = other.frameDiffs.copy()
		if other.blinks is not None:
			self.blinks = other.blinks.copy()
		if other.filteredPupilLocations is not None:
			self.filteredPupilLocations = other.filteredPupilLocations


	def FindPupils(self, endFrame = None, bilateral = None, nThreads = 0):
		"""
		Find the circles, i.e. pupils in the rawFrames, see eyetrack.video2circles()
		@param endFrame:		frame to read to, defaults to reading all rawFrames
		@param bilateral:		if present, radius to use for surface blur
		@param nThreads:		number of threads to use for finding pupils. need to be implemented
		@type endFrame:			int?
		@type bilateral:		int?
		@type nThreads:			int
		"""
		if ((endFrame is None) or endFrame > self.nFrames):
			endFrame = self.nFrames

		self.frameDiffs = numpy.r_[0, numpy.sum(numpy.diff(self.rawFrames, axis = 0) ** 2, (1, 2, 3))]
		self.blinks = numpy.where(self.frameDiffs > self.frameDiffs.mean() + self.frameDiffs.std() * 2, True, False)

		self.rawPupilLocations = numpy.zeros([self.rawFrames.shape[0], 3])
		### === parallel for ===
		if nThreads == 1:
			PupilFinder.HoughWorker(self.rawFrames, self.window, bilateral, self.blur, self.dp,
									self.minDistance, self.param1, self.param2, self._minRadius, self._maxRadius,
									self.rawPupilLocations)
		else:
			chunkSize = int(endFrame / nThreads)
			threads = []
			for thread in range(nThreads):
				start = chunkSize * thread
				end = start + chunkSize
				if thread == (nThreads - 1):
					end = endFrame
				threads.append(threading.Thread(target = PupilFinder.HoughWorker,
												args = (self.rawFrames[start:end, :, :, :], self.window, bilateral, self.blur, self.dp,
														self.minDistance, self.param1, self.param2, self._minRadius, self._maxRadius,
														self.rawPupilLocations[start:end, :])))
			for thread in threads:
				thread.start()
			for thread in threads:
				thread.join()


	def FilterPupils(self, windowSize = 15, outlierThresholds = None, filterPupilSize = True):
		"""
		Filters raw pupil locations
		@param windowSize:			median filter time window size
		@param outlierThresholds:	thresholds in percentiles at which to nan outliers, if none, does not nan outliers
		@param filterPupilSize:		filter pupil size alone with position?
		@type windowSize:			int
		@type outlierThresholds:	list<float>?
		@type filterPupilSize:		bool
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
		@param directory: 		directory to which to save
		@param startFrame:		first frame to draw
		@param endFrame: 		last frame to draw, defaults to all of them
		@param filtered:		use filtered trace instead of unfiltered?
		@param burnLocation:	burn location of pupil into image?
		@param directory: 		str
		@param startFrame:		int?
		@param endFrame: 		int?
		@param filtered:		bool
		@param burnLocation:	bool
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
		@param fileName:		file name
		@param startFrame:		first frame to draw
		@param endFrame: 		last frame to draw, defaults to all of them
		@param filtered:		use filtered trace instead of unfiltered?
		@param burnLocation:	burn location of pupil into image?
		@param fileName:		str
		@param startFrame:		int?
		@param endFrame: 		int?
		@param filtered:		bool
		@param burnLocation:	bool
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
		@param filtered:	temporally filter the traces?
		@param fps:			f present, resample the traces to this FPS
		@param filtered:	bool
		@param fps:			int?
		@return:	pupil locations
		@rtype: numpy.ndarray
		"""
		traces = self.filteredPupilLocations if filtered else self.rawPupilLocations
		if fps is None:
			return traces

		if numpy.abs(fps - self.fps) < 0.1:
			return traces

		timeScaleFactor = float(fps) / float(self.fps)
		nFramesOut = int(traces.shape[0] * timeScaleFactor)
		outTrace = numpy.zeros([nFramesOut, traces.shape[1]])
		for i in range(nFramesOut):
			start = int(i / timeScaleFactor)
			end = int((i + 1) / timeScaleFactor)
			outTrace[i, :] = traces[start:end, :].mean(0)

		outTrace = numpy.array(outTrace)

		return outTrace

	def Save(self, fileName = None, outFile = None):
		"""
		Save out information
		@param fileName: 	name of file to save, must be not none if fileObject is None
		@param outFile: 	existing object to write to
		@type fileName: 	str?
		@type outFile: 	zipfile?
		"""

		closeOnFinish = outFile is None  # we close the file only if this is the actual function that started the file

		if outFile is None:
			outFile = ZipFile(fileName, 'w')

		super(PupilFinder, self).Save(None, outFile)

		if self.rawPupilLocations is not None:
			SaveNPY(self.rawPupilLocations, outFile, 'rawPupilLocations.npy')
		if self.frameDiffs is not None:
			SaveNPY(self.frameDiffs, outFile, 'frameDiffs.npy')
		if self.blinks is not None:
			SaveNPY(self.blinks, outFile, 'blinks.npy')
		if self.filteredPupilLocations is not None:
			SaveNPY(self.filteredPupilLocations, outFile, 'filteredPupilLocations.npy')

		if closeOnFinish:
			outFile.close()


	def Load(self, fileName = None, inFile = None):
		"""
		Loads in information
		@param fileName: 	name of file to read, must not be none if infile is none
		@param inFile:		existing object to read from
		@type fileName: 	str?
		@type inFile:		zipfile?
		"""

		closeOnFinish = inFile is None
		if inFile is None:
			inFile = ZipFile(fileName, 'r')

		super(PupilFinder, self).Load(None, inFile)

		subFiles = inFile.NameToInfo.keys()
		if 'rawPupilLocations.npy' in subFiles:
			self.rawPupilLocations= ReadNPY(inFile, 'rawPupilLocations.npy')
		if 'frameDiffs.npy' in subFiles:
			self.frameDiffs = ReadNPY(inFile, 'frameDiffs.npy')
		if 'blinks.npy' in subFiles:
			self.blinks = ReadNPY(inFile, 'blinks.npy')
		if 'filteredPupilLocations.npy' in subFiles:
			self.filteredPupilLocations = ReadNPY(inFile, 'filteredPupilLocations.npy')

		if closeOnFinish:
			inFile.close()