import numpy
from zipfile import ZipFile
from EyetrackingUtilities import ReadNPY, SaveNPY, parallelize
from scipy.interpolate import Rbf as RBF
from PupilFinder import PupilFinder
from TemplatePupilFinder import TemplatePupilFinder

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
	def GeneratePoints(width = 1024, height = 768, nHorizontal = 7, nVertical = 5, DPIUnscaleFactor = 1.0):
		"""
		Generates calibration point locations. See /auto/k8/anunez/stimulusdata/eyetrack1024/draw_array_transparent_withborder.py
		or EyetrackingCalibrationHUD::EventConstruct for how these points were generated for presentation
		@param width: 				int, screen width
		@param height: 				int, screen height
		@param nHorizontal: 		int, number of horizontal points
		@param nVertical: 			int, number of vertical points
		@param DPIUnscaleFactor:	float, factor used to _unscale_ the DPI scaling in Unreal UMG, return value of UCarlaUMGBase::DPIScaleFactor
		@return: [(nHorizontal x nVertical) x 2] array
		"""

		Yspace = int(height / nVertical / DPIUnscaleFactor)
		Xspace = int(width / nHorizontal / DPIUnscaleFactor)

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
				 calibrationDuration = 2, calibrationDelay = 2, templates = True):
		"""
		Constructor
		@param calibrationVideoFile:	str, name of video file
		@param calibrationBeginTime:	4ple<int>?, time of calibration sequence onset
		@param calibrationPositions:	[n x 2] array?, pixel positions of calibration points
		@param calibrationOrder:		array<int>?, sequence in which points were presented
		@param calibrationDuration:		float, duration of fixation time per point
		@param calibrationDelay:		float, delay in seconds from begin time to first fixation
		@param templates:				bool, use template matching instead of hough circles?
		"""
		self.calibrationVideoFile = calibrationVideoFile
		# self.timestampReader = VideoTimestampReader(calibrationVideoFile)
		# self.timestampReader.ParseTimestamps()
		if templates:
			self.pupilFinder = TemplatePupilFinder(calibrationVideoFile)
		else:
			self.pupilFinder = PupilFinder(calibrationVideoFile)

		self.hasGlint = templates

		self.calibrationBeginTime = calibrationBeginTime
		self.calibrationPositions = calibrationPositions if calibrationPositions is not None else EyetrackingCalibrator.GeneratePoints()
		self.calibrationOrder = calibrationOrder if calibrationOrder is not None else EyetrackingCalibrator.CalibrationOrder35
		self.calibrationDuration = calibrationDuration
		self.calibrationDelay = calibrationDelay

		# used for searching through different durations
		self.bestDuration = 0
		self.bestDurationError = 100

		# because the calibration points get re-ordered
		self.initialCalibrationPositions = self.calibrationPositions.copy()
		self.initialCalibrationOrder = numpy.array(self.calibrationOrder)

		self.pupilCalibrationPositions = None		# mean/median pupil positions in video frames for each calibration point
		self.pupilCalibrationVariances = None		# Variance in the calibration point eye positions

		self.glintCalibrationPositions = None
		self.glintCalibrationVariances = None

		self.bestSmoothness = None				# float, best smoothness for the interpolater
		self.bestMethod = None					# str, best method for interpolating
		self.bestError = -1						# float, error on best fit
		self.horizontalInterpolater = None		# RBF, final horizontal interpolater
		self.verticalInterpolater = None		# RBF, final vertical interpolater


	def Copy(self, duration = None, delay = None):
		"""
		Make a copy of this object, but does not copy the pupil finder's frames so we save space.
		@param duration:	float?, fixation duration to use for the new object, if None will copy this
		@param delay:		float?, fixation delay to use for the new object, if None will copy this
		@return: EyetrackingCalibrator
		"""
		newCalibrator = EyetrackingCalibrator(None, self.calibrationBeginTime, self.initialCalibrationPositions, self.initialCalibrationOrder,
											  duration if duration is not None else self.calibrationDuration,
											  delay if delay is not None else self.calibrationDelay, True)
		newCalibrator.pupilFinder.InitFromOther(self.pupilFinder)
		return newCalibrator


	def FindPupils(self, window = None, blur = 5, dp = 1, minDistance = 600, param1 = 80,
				   param2 = 20, minRadius = 15, maxRadius = 22, windowSize = 15, outlierThresholds = None,
				   filterPupilSize = True, surfaceBlur = None, nThreads = None):
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
		@param surfaceBlur:			int?, if present, radius to use for surface blur
		@param nThreads:			int?, number of threads to use for pupil finding. if none, use all cores
		@return:
		"""
		# self.pupilFinder = PupilFinder(None, window, blur, dp, minDistance, param1, param2, minRadius, maxRadius, self.pupilFinder)

		if nThreads is None:
			import multiprocessing
			nThreads = multiprocessing.cpu_count()
		self.pupilFinder.ParseTimestamps(nThreads)
		self.pupilFinder.window = window
		self.pupilFinder.blur = blur
		self.pupilFinder.dp = dp
		self.pupilFinder.minDistance = minDistance
		self.pupilFinder.param1 = param1
		self.pupilFinder.param2 = param2
		self.pupilFinder.minRadius = minRadius
		self.pupilFinder.maxRadius = maxRadius
		self.pupilFinder.FindPupils(bilateral = surfaceBlur, nThreads = nThreads)
		self.pupilFinder.FilterPupils(windowSize, outlierThresholds, filterPupilSize)


	def EstimateCalibrationPointPositions(self, beginTime = None, method = numpy.nanmedian, startDelay = 1.0 / 6.0, endSkip = 0.0, duration = None,
										  filtered = True, verbose = True):
		"""
		Estimates the pupil positions corresponding to each calibration point
		@param beginTime:		4ple<int>?, time of calibration sequence onset	TODO: replace this with a frame number
		@param method:			function, method used to aggregated points to one summary point
		@param startDelay:		float, time delay in seconds to account for eye movement delay
		@param endSkip:			float, time in seconds to clip from the end of a fixation period
		@param duration:		float|list<float>?, duration of each fixation point, if list, search through all and select best
		@param filtered:		bool, use filtered pupil locations?
		@param verbose:			bool, print stuff?
		@return:
		"""
		if (self.pupilFinder is None):
			self.FindPupils()

		if (beginTime is None):
			beginTime = self.calibrationBeginTime

		if (duration is None):
			duration = self.calibrationDuration

		isList = False
		try:
			isList = len(duration)
		except:
			pass

		if isList:	# see this is where overloading comes in useful
			self.bestDurationError = 1000
			self.bestDuration = 0
			for i in range(len(duration)):
				self.EstimateCalibrationPointPositions(beginTime, method, startDelay, endSkip, duration[i], filtered)
				if self.bestError < self.bestDurationError:
					self.bestDurationError = self.bestError
					self.bestDuration = duration[i]
			self.EstimateCalibrationPointPositions(beginTime, method, startDelay, endSkip, self.bestDuration, filtered)
		else:
			firstFrame = self.pupilFinder.FindOnsetFrame(beginTime[0], beginTime[1], beginTime[2], beginTime[3])
			firstFrame += int(self.calibrationDelay * self.pupilFinder.fps)
			startOffset = int(self.pupilFinder.fps * startDelay)	# convert startDelay time to frame counts
			endOffset = int(self.pupilFinder.fps * endSkip)  # convert startDelay time to frame counts

			pupilPosition = []
			pupilVariance = []
			if self.hasGlint:
				glintPosition = []
				glintVariance = []
			for point in range(len(self.calibrationOrder)):
				start = int(point * duration * self.pupilFinder.fps + firstFrame + startOffset)
				end = int((point + 1) * duration * self.pupilFinder.fps + firstFrame - endOffset)
				if filtered:
					pupilPosition.append(method(self.pupilFinder.filteredPupilLocations[start:end, :], 0))
					pupilVariance.append(numpy.nanstd(self.pupilFinder.filteredPupilLocations[start:end, :], 0))
					if self.hasGlint:
						glintPosition.append(method(self.pupilFinder.filteredGlintLocations[start:end, :], 0))
						glintVariance.append(numpy.nanstd(self.pupilFinder.filteredGlintLocations[start:end, :], 0))
				else:
					pupilPosition.append(method(self.pupilFinder.rawPupilLocations[start:end, :], 0))
					pupilVariance.append(numpy.nanstd(self.pupilFinder.rawPupilLocations[start:end, :], 0))
					if self.hasGlint:
						glintPosition.append(method(self.pupilFinder.rawGlintLocations[start:end, :], 0))
						glintVariance.append(numpy.nanstd(self.pupilFinder.rawGlintLocations[start:end, :], 0))
			self.pupilCalibrationPositions = numpy.asarray(pupilPosition)
			self.pupilCalibrationVariances = numpy.asarray(pupilVariance)
			if self.hasGlint:
				self.glintCalibrationPositions = numpy.asarray(glintPosition)
				self.glintCalibrationVariances = numpy.asarray(glintVariance)

			valid = numpy.isfinite(self.pupilCalibrationPositions[:, :2].sum(1))
			if numpy.any(numpy.logical_not(valid)):	# if there are nan points, which are bad
				calibrationOrder = []
				for i in range(len(self.initialCalibrationOrder)):
					if valid[i]:
						calibrationOrder.append(self.initialCalibrationOrder[i])
					else:
						if verbose:
							print('Skipping bad calibration point {}'.format(self.calibrationOrder[i]))
				self.calibrationOrder = calibrationOrder
				self.pupilCalibrationPositions = self.pupilCalibrationPositions[valid, :]
				self.pupilCalibrationVariances = self.pupilCalibrationVariances[valid, :]
				if self.hasGlint:
					self.glintCalibrationPositions = self.glintCalibrationPositions[valid, :]
					self.glintCalibrationVariances = self.glintCalibrationVariances[valid, :]

			# reorder the calibration points to be the same as the calibration sequence
			points = []
			for i in self.calibrationOrder:
				points.append(self.initialCalibrationPositions[i, :])
			self.calibrationPositions = numpy.asarray(points)


	def Fit(self, smoothnesses = numpy.linspace(-0.001, 10, 100), methods = ['thin-plate', 'multiquadric', 'linear', 'cubic'], varianceThreshold = None,
			glintVector = False, searchThreshold = 40):
		"""
		Fit an interpolator to the points via LOO x-validation
		@param smoothnesses:		list<float>, smoothnesses to try for interpolations
		@param methods:				list<str>, methods to try
		@param varianceThreshold:	float?, threshold of variance in the calibration positions to throw away
		@param glintVector:			bool, use pupil-glint vector instead of just the pupil position?
		@param searchThreshold:		float, if the error is above this threshold, search over delays/durations. will not search if is 0
		@return:
		"""
		if not self.hasGlint:
			glintVector = False

		valid = numpy.ones(self.pupilCalibrationVariances.shape[0]).astype(bool)
		if varianceThreshold:
			xThreshold = numpy.percentile(self.pupilCalibrationVariances[:, 0], varianceThreshold)
			yThreshold = numpy.percentile(self.pupilCalibrationVariances[:, 1], varianceThreshold)

			valid = self.pupilCalibrationVariances[:, 0] <= xThreshold
			valid[self.pupilCalibrationVariances[:, 1] > yThreshold] = False

		eyeCalibrationPositions = self.pupilCalibrationPositions[valid, :]
		if glintVector:	# vector from glint to pupil
			eyeCalibrationPositions -= self.glintCalibrationPositions[valid, :]

		calibrationPositions = self.calibrationPositions[valid, :]
		calibrationOrder = []
		for i in range(valid.shape[0]):
			if valid[i]:
				calibrationOrder.append(self.calibrationOrder[i])

		def LeaveOneOutXval(smoothness, method):
			"""
			Leave on out estimation, returns RMS deviation from actual points
			@param smoothness:
			@param method:
			@return:
			"""
			estimates = numpy.zeros([len(calibrationOrder), 2])
			for i in range(len(calibrationOrder)):
				fit = [True] * len(calibrationOrder)
				fit[i] = False

				horizontal = RBF(eyeCalibrationPositions[fit, 0], eyeCalibrationPositions[fit, 1],
								 calibrationPositions[fit, 0], function = method, smooth = smoothness)

				vertical = RBF(eyeCalibrationPositions[fit, 0], eyeCalibrationPositions[fit, 1],
							   calibrationPositions[fit, 1], function = method, smooth = smoothness)

				estimates[i, :] = [horizontal(eyeCalibrationPositions[i, 0], eyeCalibrationPositions[i, 1]),
								vertical(eyeCalibrationPositions[i, 0], eyeCalibrationPositions[i, 1])]

			return numpy.sqrt(numpy.mean((estimates - calibrationPositions) ** 2))

		errors = numpy.zeros([len(smoothnesses), len(methods)])
		for s in range(len(smoothnesses)):
			for m in range(len(methods)):
				errors[s, m] = LeaveOneOutXval(smoothnesses[s], methods[m])

		s, m = numpy.unravel_index(errors.argmin(), errors.shape)
		self.bestSmoothness = smoothnesses[s]
		self.bestMethod = methods[m]
		self.bestError = errors[s, m]

		self.horizontalInterpolater = RBF(eyeCalibrationPositions[:, 0], eyeCalibrationPositions[:, 1],
										  calibrationPositions[:, 0], function = self.bestMethod, smooth = self.bestSmoothness)

		self.verticalInterpolater  =  RBF(eyeCalibrationPositions[:, 0], eyeCalibrationPositions[:, 1],
										  calibrationPositions[:, 1], function = self.bestMethod, smooth = self.bestSmoothness)

		if (searchThreshold > 0) and (self.bestError > searchThreshold):
			print('Min error {:.2f} is above threshold of {:.0f} and will be searching'.format(self.bestError, searchThreshold))
			self.SearchAndFit()


	def SearchAndFit(self, durations = numpy.arange(1.98, 2.21, 0.01), delays = numpy.arange(1.98, 2.41, 0.01), verbose = False):
		"""
		Searches for best duration and delay
		@param durations: 	list<float>, list of durations to search over
		@param delays: 		list<float>, list of delay lengths to search over
		@param verbose: 	bool
		@return:	tuple<float, float, float>, best duration, delay, and error
		"""

		def singleCombo(delayAndDuration):
			delay = delayAndDuration[0]
			duration = delayAndDuration[1]
			calibrator = self.Copy(duration, delay)
			calibrator.EstimateCalibrationPointPositions(verbose = verbose)
			calibrator.Fit(searchThreshold = 0)
			return calibrator.bestError

		delayAndDurations = []
		for delay in delays:
			for duration in durations:
				delayAndDurations.append((delay, duration))

		errors = parallelize(singleCombo, delayAndDurations)
		minError = 1000
		minIndex = None
		for i in range(len(delayAndDurations)):
			if errors[i] < minError:
				minError = errors[i]
				minIndex = i

		self.calibrationDelay = delayAndDurations[minIndex][0]
		self.calibrationDuration = delayAndDurations[minIndex][1]
		self.EstimateCalibrationPointPositions(verbose = verbose)
		self.Fit(searchThreshold = 0)

		return (delayAndDurations[minIndex][1], delayAndDurations[minIndex][0], errors[minIndex])


	def TransformToScreenCoordinates(self, pupilFinder = None, trace = None, videoFileName = None, replaceNanWithPrevious = True):
		"""
		Transforms an eyetracking video file or pupil finder to screen coords using this calibration.
		It's better to give a pupilFinder object that uses customized parameters than to use the defaults and use a file
		@param pupilFinder:				PupilFinder?, pupil finder object that has been run
		@param trace:					[n x 3] array?, output traces from something
		@param videoFileName:			str?, file to use
		@param replaceNanWithPrevious:	bool, replace nan values with the previous value?
		@return:
		"""

		# grumble grumble function overloading...
		isNone = [pupilFinder is not None, trace is not None, videoFileName is not None]
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

		transformed = numpy.zeros([trace.shape[0], 2])
		transformed[:, 0] = self.horizontalInterpolater(trace[:, 0], trace[:, 1])
		transformed[:, 1] = self.verticalInterpolater(trace[:, 0], trace[:, 1])

		if (replaceNanWithPrevious):
			for i in range(transformed.shape[0] - 1):
				if numpy.any(numpy.isnan(transformed[i + 1, :])):
					transformed[i + 1, :] = transformed[i, :]

		return transformed

	def Save(self, fileName):
		"""
		Saves calibrated interpolators and whatever else
		@param fileName:
		@return:
		"""

		outFile = ZipFile(fileName, 'w')

		self.pupilFinder.Save(None, outFile)

		if self.pupilCalibrationPositions is not None:
			SaveNPY(self.pupilCalibrationPositions, outFile, 'pupilCalibrationPositions.npy')
		if self.pupilCalibrationVariances is not None:
			SaveNPY(self.pupilCalibrationVariances, outFile, 'pupilCalibrationVariances.npy')
		if self.glintCalibrationPositions is not None:
			SaveNPY(self.glintCalibrationPositions, outFile, 'glintCalibrationPositions.npy')
		if self.glintCalibrationVariances is not None:
			SaveNPY(self.glintCalibrationVariances, outFile, 'glintCalibrationVariances.npy')
		outFile.close()

		# rbfs can't be save right now
		# if self.horizontalInterpolater is not None:
		# 	SavePickle(self.horizontalInterpolater, outFile, 'horizontalInterpolator.pkl')
		# if self.verticalInterpolater is not None:
		# 	SavePickle(self.verticalInterpolater, outFile, 'verticalInterpolator.pkl')


	def Load(self, fileName):
		"""
		Load previously saved interpolators
		@param fileName:
		@return:
		"""

		inFile = ZipFile(fileName, 'r')

		self.pupilFinder.Load(None, inFile)

		subFiles = inFile.NameToInfo.keys()
		if 'pupilCalibrationPositions.npy' in subFiles:
			self.pupilCalibrationPositions = ReadNPY(inFile, 'pupilCalibrationPositions.npy')
		if 'pupilCalibrationVariances.npy' in subFiles:
			self.pupilCalibrationVariances = ReadNPY(inFile, 'pupilCalibrationVariances.npy')
		if 'glintCalibrationPositions.npy' in subFiles:
			self.glintCalibrationPositions = ReadNPY(inFile, 'glintCalibrationPositions.npy')
		if 'glintCalibrationVariances.npy' in subFiles:
			self.glintCalibrationVariances = ReadNPY(inFile, 'glintCalibrationVariances.npy')

		inFile.close()
		# if 'horizontalInterpolator.pkl' in subFiles:
		# 	self.horizontalInterpolater = ReadPickle(inFile, 'horizontalInterpolator.pkl')
		# if 'verticalInterpolator.pkl' in subFiles:
		# 	self.verticalInterpolater = ReadPickle(inFile, 'verticalInterpolator.pkl')
