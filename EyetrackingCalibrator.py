import numpy
from scipy.interpolate import Rbf as RBF
from VideoTimestampReader import VideoTimestampReader
from PupilFinder import PupilFinder

class EyetrackingCalibrator(object):
	"""
	Calibrates eyetracking from a raw video. Estimates a TPS warping for the points.
	"""

	# order used in 1024x768 single repeat calibration (e.g. driving auto calibration)
	CalibrationOrder35 = [13, 30, 17, 16, 2, 27, 1, 28, 25, 10, 26, 9, 14, 5, 34, 32,
						  31, 12, 8, 33, 18, 19, 3, 23, 29, 20, 7, 0, 4, 24, 22, 11, 15,
						  21, 6]

	# order used in 1024x768 two repeat calibration (e.g. stimulus/eyetracking/play.py)
	# see stimulus/eyetracking/eyetrack.index
	# these numbers are offset by -1 b/c image 0 there is a blank screen
	CalibrationOrder70 = [13, 30, 17, 16, 2, 27, 1, 28, 25, 10, 26, 9, 14, 5, 34, 32,
						  31, 12, 8, 33, 18, 19, 3, 23, 29, 20, 7, 0, 4, 24, 22, 11, 15,
						  21, 6, 30, 6, 15, 33, 32, 10, 20, 0, 7, 19, 8, 21, 1, 11, 4, 24,
						  13, 18, 34, 26, 17, 14, 5, 27, 25, 28, 22, 31, 16, 29, 9, 23, 2,
						  3, 12]

	@staticmethod
	def GeneratePoints(width = 1024, height = 768, nHorizontal = 7, nVertical = 5):
		"""
		Generates calibration point locations. See /auto/k8/anunez/stimulusdata/eyetrack1024/draw_array_transparent_withborder.py
		or EyetrackingCalibrationHUD::EventConstruct for how these points were generated for presentation
		@param width: 			int, screen width
		@param height: 			int, screen height
		@param nHorizontal: 	int, number of horizontal points
		@param nVertical: 		int, number of vertical points
		@return: [(nHorizontal x nVertical) x 2] array
		"""

		Yspace = int(height / nVertical)
		Xspace = int(width / nHorizontal)

		points = numpy.zeros([nHorizontal * nVertical, 2], dtype = int)
		index = 0
		for i in range(-nHorizontal / 2 + 1, nHorizontal / 2 + 1):
			x = width / 2 + i * Xspace
			for j in range(-nVertical / 2 + 1, nVertical / 2 + 1):
				y = height / 2 + j * Yspace
				points[index, :] = [x, y]
				index += 1
		return points


	def __init__(self, calibrationVideoFile, calibrationBeginTime = None, calibrationPositions = None, calibrationOrder = None,
				 calibrationDuration = 2):
		"""
		Constructor
		@param calibrationVideoFile:	str, name of video file
		@param calibrationBeginTime:	4ple<int>?, time of calibration sequence onset
		@param calibrationPositions:	[n x 2] array?, pixel positions of calibration points
		@param calibrationOrder:		array<int>?, sequence in which points were presented
		@param calibrationDuration:		float, duration of fixation time per point
		"""
		self.calibrationVideoFile = calibrationVideoFile
		self.timestampReader = VideoTimestampReader(calibrationVideoFile)
		self.timestampReader.ParseTimestamps()
		self.pupilFinder = None

		self.calibrationBeginTime = calibrationBeginTime
		self.calibrationPositions = calibrationPositions if calibrationPositions else EyetrackingCalibrator.GeneratePoints()
		self.calibrationOrder = calibrationOrder if calibrationOrder else EyetrackingCalibrator.CalibrationOrder35
		self.calibrationDuration = calibrationDuration

		self.eyeCalibrationPositions = None		# mean/median eye positions in video frames for each calibration point

		self.bestSmoothness = None				# float, best smoothness for the interpolater
		self.bestMethod = None					# str, best method for interpolating
		self.horizontalInterpolater = None		# RBF, final horizontal interpolater
		self.verticalInterpolater = None		# RBF, final vertical interpolater


	def FindPupils(self, window = None, blur = 5, dp = 1, minDistance = 600, param1 = 80,
				   param2 = 20, minRadius = 20, maxRadius = 0, windowSize = 15, outlierThresholds = None, filterPupilSize = True):
		"""
		Finds pupil traces
		@param window: 				4-ple<int>?, subwindow in frame to examine, order [left, right, top, bottom]
		@param blur: 				int, median blur filter width
		@param dp: 					float, inverse ratio of accumulator resolution to image resolution
		@param minDistance: 		float, min distance between centers of detected circles
		@param param1: 				float, higher threshold for canny edge detector
		@param param2: 				float, accumulator threshold at detection stage, smaller => more errors
		@param minRadius: 			int, min circle radius
		@param maxRadius: 			int, max circle radius
		@param windowSize:			int, median filter time window size
		@param outlierThresholds:	list<float>?, thresholds in percentiles at which to nan outliers, if none, does not nan outliers
		@param filterPupilSize:		bool, filter pupil size alone with position?
		@return:
		"""
		self.pupilFinder = PupilFinder(None, window, blur, dp, minDistance, param1, param2, minRadius, maxRadius, self.timestampReader)
		self.pupilFinder.FindPupils()
		self.pupilFinder.FilterPupils(windowSize, outlierThresholds, filterPupilSize)


	def EstimateCalibrationPointPositions(self, beginTime = None, method = numpy.nanmedian, delay = 1.0 / 6.0, duration = None):
		"""
		Estimates the pupil positions corresponding to each calibration point
		@param beginTime:	4ple<int>?, time of calibration sequence onset
		@param method:		function, method used to aggregated points to one summary point
		@param delay:		float, time delay in seconds to account for eye movement delay
		@param duration:	float?, duration of each fixation point
		@return:
		"""
		if (self.pupilFinder is None):
			self.FindPupils()

		if (beginTime is None):
			beginTime = self.calibrationBeginTime

		if (duration is None):
			duration = self.calibrationDuration

		firstFrame = self.timestampReader.FindOnsetFrame(beginTime[0], beginTime[1], beginTime[2], beginTime[3])
		offset = int(self.timestampReader.fps * delay)	# convert delay time to frame counts

		eyePosition = []
		for point in range(len(self.calibrationOrder)):
			start = int(point * duration * self.timestampReader.fps + firstFrame + offset)
			end = int((point + 1) * duration * self.timestampReader.fps + firstFrame)
			eyePosition.append(method(self.pupilFinder.filteredPupilLocations[start:end, :], 0))
		self.eyeCalibrationPositions = numpy.asarray(eyePosition)

		valid = numpy.isfinite(self.eyeCalibrationPositions[:, :2].sum(1))
		if numpy.any(numpy.logical_not(valid)):	# if there are nan points, which are bad
			calibrationOrder = []
			for i in range(len(self.calibrationOrder)):
				if valid[i]:
					calibrationOrder.append(self.calibrationOrder[i])
				else:
					print('Skipping bad calibration point {}'.format(self.calibrationOrder[i]))
			self.calibrationOrder = calibrationOrder
			self.eyeCalibrationPositions = self.eyeCalibrationPositions[valid, :]

		# reorder the calibration points to be the same as the calibration sequence
		points = []
		for i in self.calibrationOrder:
			points.append(self.calibrationPositions[i, :])
		self.calibrationPositions = numpy.asarray(points)


	def Fit(self, smoothnesses = numpy.linspace(-0.001, 10, 100), methods = ['thin-plate', 'multiquadric', 'linear', 'cubic']):
		"""
		Fit an interpolator to the points via LOO x-validation
		@param smoothnesses:	list<float>, smoothnesses to try for interpolations
		@param methods:			list<str>, methods to try
		@return:
		"""

		def LeaveOneOutXval(smoothness, method):
			"""
			Leave on out estimation, returns RMS deviation from actual points
			@param smoothness:
			@param method:
			@return:
			"""
			estimates = numpy.zeros([len(self.calibrationOrder), 2])
			for i in range(len(self.calibrationOrder)):
				fit = [True] * len(self.calibrationOrder)
				fit[i] = False

				horizontal = RBF(self.eyeCalibrationPositions[fit, 0], self.eyeCalibrationPositions[fit, 1],
								 self.calibrationPositions[fit, 0], function = method, smooth = smoothness)

				vertical = RBF(self.eyeCalibrationPositions[fit, 0], self.eyeCalibrationPositions[fit, 1],
							   self.calibrationPositions[fit, 1], function = method, smooth = smoothness)

				estimates[i, :] = [horizontal(self.eyeCalibrationPositions[i, 0], self.eyeCalibrationPositions[i, 1]),
								vertical(self.eyeCalibrationPositions[i, 0], self.eyeCalibrationPositions[i, 1])]

			return numpy.sqrt(numpy.mean((estimates - self.calibrationPositions) ** 2))

		errors = numpy.zeros([len(smoothnesses), len(methods)])
		for s in range(len(smoothnesses)):
			for m in range(len(methods)):
				errors[s, m] = LeaveOneOutXval(smoothnesses[s], methods[m])

		s, m = numpy.unravel_index(errors.argmin(), errors.shape)
		self.bestSmoothness = smoothnesses[s]
		self.bestMethod = methods[m]

		self.horizontalInterpolater = RBF(self.eyeCalibrationPositions[:, 0], self.eyeCalibrationPositions[:, 1],
										  self.calibrationPositions[:, 0], function = self.bestMethod, smooth = self.bestSmoothness)

		self.verticalInterpolater  =  RBF(self.eyeCalibrationPositions[:, 0], self.eyeCalibrationPositions[:, 1],
										  self.calibrationPositions[:, 1], function = self.bestMethod, smooth = self.bestSmoothness)


	def TransformToScreenCoordinates(self, pupilFinder = None, trace = None, videoFileName = None):
		"""
		Transforms an eyetracking video file or pupil finder to screen coords using this calibration.
		It's better to give a pupilFinder object that uses customized parameters than to use the defaults and use a file
		@param pupilFinder:			PupilFinder?, pupil finder object that has been run
		@param trace:				[n x 3] array?, output traces
		@param videoFileName:		str?, file to use
		@return:
		"""

		# grumble grumble function overloading...
		isNone = [pupilFinder is None, trace is None, videoFileName is None]
		if (sum(isNone) != 1):
			print('Check your input. there can be only one!')
			return

		# ... or switch-case ...
		if (videoFileName is not None):
			pupilFinder = PupilFinder(videoFileName)
			pupilFinder.FindPupils()
			pupilFinder.FilterPupils()
		if (pupilFinder is not None):
			trace = pupilFinder.filteredPupilLocations

		transformed = numpy.zeros(trace.shape)
		transformed[:, 0] = self.horizontalInterpolater(trace[:, 0], trace[:, 1])
		transformed[:, 1] = self.verticalInterpolater(trace[:, 0], trace[:, 1])

		return transformed