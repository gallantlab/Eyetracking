import numpy
from enum import IntEnum
from zipfile import ZipFile
from .EyetrackingUtilities import ReadNPY, SaveNPY, parallelize
from scipy.interpolate import Rbf as RBF
from .PupilFinder import PupilFinder
from .TemplatePupilFinder import TemplatePupilFinder
from .AvotecPupilFinder import AvotecPupilFinder
import sys
PYTHON_VERSION = sys.version_info.major


class PupilFindingMethod(IntEnum):
	"""
	Enum for pupil finding method. Determines which pupil finder class to use
	"""
	Hough = 0		# Hough transform to find an ellipse
	Templates = 1	# Use template matching for a black circle on a white background
	Avotec = 2		# Use avotec output file


class EyetrackingCalibrator(object):
	"""
	Calibrates eyetracking from a raw video. Estimates a TPS warping for the points.
	"""

	CalibrationOrder35 = [13, 30, 17, 16, 2, 27, 1, 28, 25, 10, 26, 9, 14, 5, 34, 32,
						  31, 12, 8, 33, 18, 19, 3, 23, 29, 20, 7, 0, 4, 24, 22, 11, 15,
						  21, 6]
	"""
	@cvar: order used in 1024x768 single repeat calibration (e.g. driving auto calibration)
	"""

	CalibrationOrder70 = [13, 30, 17, 16, 2, 27, 1, 28, 25, 10, 26, 9, 14, 5, 34, 32,
						  31, 12, 8, 33, 18, 19, 3, 23, 29, 20, 7, 0, 4, 24, 22, 11, 15,
						  21, 6, 30, 6, 15, 33, 32, 10, 20, 0, 7, 19, 8, 21, 1, 11, 4, 24,
						  13, 18, 34, 26, 17, 14, 5, 27, 25, 28, 22, 31, 16, 29, 9, 23, 2,
						  3, 12]
	"""
	@cvar: order used in 1024x768 two repeat calibration (e.g. stimulus/eyetracking/play.py)
	see stimulus/eyetracking/eyetrack.index in stimulus marchine
	these numbers are offset by -1 b/c image 0 there is a blank screen
	"""

	@staticmethod
	def GeneratePoints(width = 1024, height = 768, nHorizontal = 7, nVertical = 5, DPIUnscaleFactor = 1.0):
		"""
		Generates calibration point locations. See /auto/k8/anunez/stimulusdata/eyetrack1024/draw_array_transparent_withborder.py
		or EyetrackingCalibrationHUD::EventConstruct for how these points were generated for presentation
		@param width: 				screen width
		@type width:				int
		@param height: 				screen height
		@type height:				int
		@param nHorizontal: 		number of horizontal points
		@type nHorizontal:			int
		@param nVertical: 			number of vertical points
		@type nVertical:			int
		@param DPIUnscaleFactor:	factor used to _unscale_ the DPI scaling in Unreal UMG, return value of UCarlaUMGBase::DPIScaleFactor
		@type DPIUnscaleFactor:		float
		@return: [(nHorizontal x nVertical) x 2] array
		@rtype: numpy.ndarray
		"""

		Yspace = int(height / nVertical / DPIUnscaleFactor)
		Xspace = int(width / nHorizontal / DPIUnscaleFactor)

		xStart = int(- nHorizontal / 2)
		yStart = int(-nVertical / 2)
		if (PYTHON_VERSION < 3):
			xStart += 1
			yStart += 1

		points = numpy.zeros([nHorizontal * nVertical, 2], dtype = int)
		index = 0
		for i in range(xStart, int(nHorizontal / 2) + 1):
			x = width / 2 + i * Xspace
			for j in range(yStart, int(nVertical / 2) + 1):
				y = height / 2 + j * Yspace
				points[index, :] = [x, y]
				index += 1
		return points


	def __init__(self, calibrationVideoFile, calibrationBeginTime = None, calibrationPositions = None, calibrationOrder = None,
				 calibrationDuration = 2, calibrationDelay = 2, method = PupilFindingMethod.Templates):
		"""
		Constructor
		@param calibrationVideoFile:	name of video file or avotec file
		@type calibrationVideoFile:		str
		@param calibrationBeginTime:	time of calibration sequence onset
		@type calibrationBeginTime:		4ple<int>?
		@param calibrationPositions:	pixel positions of calibration points
		@type calibrationPositions:		[n x 2] array?
		@param calibrationOrder:		sequence in which points were presented
		@type calibrationOrder:			array<int>?
		@param calibrationDuration:		duration of fixation time per point
		@type calibrationDuration:		float
		@param calibrationDelay:		delay in seconds from begin time to first fixation
		@type calibrationDelay:			float
		@param method:					what method to use to find the pupil
		@type method:					PupilFindingMethod
		"""
		self.calibrationVideoFile = calibrationVideoFile
		"""
		@ivar: video file name
		@type: str
		"""
		# self.timestampReader = VideoTimestampReader(calibrationVideoFile)
		# self.timestampReader.ParseTimestamps()
		if method == PupilFindingMethod.Templates:
			self.pupilFinder = TemplatePupilFinder(calibrationVideoFile)
			"""
			@ivar: pupil finder object used to find calibration pupils
			@type: PupilFinder
			"""
		elif method == PupilFindingMethod.Hough:
			self.pupilFinder = PupilFinder(calibrationVideoFile)
		elif method == PupilFindingMethod.Avotec:
			self.pupilFinder = AvotecPupilFinder(calibrationVideoFile)
		else:
			raise ValueError('Unknown pupil finding method: {}'.format(method))

		self.hasGlint = method == PupilFindingMethod.Templates
		"""
		@ivar: can this object find the glint in addition to the pupil?
		@type: bool
		"""

		self.calibrationBeginTime = calibrationBeginTime
		"""
		@ivar: Timestamp of first TTL in calibration
		@type: tuple<int, int, int, int>
		"""
		self.calibrationPositions = calibrationPositions if calibrationPositions is not None else EyetrackingCalibrator.GeneratePoints()
		"""
		@ivar: screen pixel positions of calibration dots
		@type: numpy.ndarray
		"""
		self.calibrationOrder = calibrationOrder if calibrationOrder is not None else EyetrackingCalibrator.CalibrationOrder35
		"""
		@ivar: order in which the calibration dots are presented
		@type: list<int>
		"""
		self.calibrationDuration = calibrationDuration
		"""
		@ivar: time in seconds each dot is presented
		@type: float
		"""
		self.calibrationDelay = calibrationDelay
		"""
		@ivar: time in seconds between the first TTL and presentation of first dot
		@type: float
		"""

		# used for searching through different durations
		self.bestDuration = 0
		"""
		@ivar: If searching for different calibrationDurations, best duration found
		@type: float
		"""
		self.bestDurationError = 100
		"""
		@ivar: If searching for different calibrationDurections, error corresponding to best duration
		@type: float
		"""

		# because the calibration points get re-ordered
		self.initialCalibrationPositions = self.calibrationPositions.copy()
		"""
		@ivar: Stored copy of calibration positions, because calibrationPositions will be shuffled according to presentation order
		@type: numpy.ndarray
		"""
		self.initialCalibrationOrder = numpy.array(self.calibrationOrder)
		"""
		@ivar: Stored copy of calibration order
		@type: list<int>
		"""

		self.pupilCalibrationPositions = None		# mean/median pupil positions in video frames for each calibration point
		"""
		@ivar: mean or median pupil positions in video positions for each calibration point
		@type: numpy.ndarray
		"""
		self.pupilCalibrationVariances = None		# Variance in the calibration point eye positions
		"""
		@ivar: variance in calibration point eye positions
		@type: numpy.ndarray
		"""

		self.glintCalibrationPositions = None
		"""
		@ivar: mean or median glint locations for each calibration point
		@type: numpy.ndarray
		"""
		self.glintCalibrationVariances = None
		"""
		@ivar: variance in glint locations for each calibration point
		@type: numpy.ndarray
		"""

		self.bestSmoothness = None				# float, best smoothness for the interpolater
		"""
		@ivar: best smootheness value for interpolating gaze location
		@type: float
		"""
		self.bestMethod = None					# str, best method for interpolating
		"""
		@ivar: best method for interpolating gaze location
		@type: str
		"""
		self.bestError = -1						# float, error on best fit
		"""
		@ivar: best leave-one-out error for gaze interpolation fitting
		@type: float
		"""
		self.horizontalInterpolater = None		# RBF, final horizontal interpolater
		"""
		@ivar: final interpolator for X gaze location
		@type: scipy.interpolate.Rbf
		"""
		self.verticalInterpolater = None		# RBF, final vertical interpolater
		"""
		@ivar: final interpolator for Y gaze location
		@type: scipy.interpolate.Rbf
		"""


	def Copy(self, duration = None, delay = None):
		"""
		Make a copy of this object, but does not copy the pupil finder's frames so we save space.
		@param duration:	float?, fixation duration to use for the new object, if None will copy this
		@param delay:		float?, fixation delay to use for the new object, if None will copy this
		@return: EyetrackingCalibrator
		@rtype: EyetrackingCalibrator
		"""
		newCalibrator = EyetrackingCalibrator(None, self.calibrationBeginTime, self.initialCalibrationPositions, self.initialCalibrationOrder,
											  duration if duration is not None else self.calibrationDuration,
											  delay if delay is not None else self.calibrationDelay, True)
		newCalibrator.pupilFinder.InitFromOther(self.pupilFinder)
		return newCalibrator


	def FindPupils(self, window = None, blur = 5, dp = 1, minDistance = 600, param1 = None,
				   param2 = None, minRadius = 10, maxRadius = 18, windowSize = 15, outlierThresholds = None,
				   filterPupilSize = True, surfaceBlur = None, nThreads = None):
		"""
		Finds pupil traces
		@param window: 				subwindow in frame to examine, order [left, right, top, bottom]
		@param blur: 				median blur filter width
		@param dp: 					inverse ratio of accumulator resolution to image resolution
		@param minDistance: 		min distance between centers of detected circles
		@param param1: 				higher threshold for canny edge detector, or sigmas below normal for blink detection in templates
		@param param2: 				accumulator threshold at detection stage, smaller => more errors
		@param minRadius: 			min circle radius
		@param maxRadius: 			max circle radius
		@param windowSize:			median filter time window size
		@param outlierThresholds:	thresholds in percentiles at which to nan outliers, if none, does not nan outliers
		@param filterPupilSize:		filter pupil size alone with position?
		@param surfaceBlur:			if present, radius to use for surface blur
		@param nThreads:			number of threads to use for pupil finding. if none, use all cores
		@type window: 				4-ple<int>?
		@type blur: 				int
		@type dp: 					float
		@type minDistance: 			float
		@type param1: 				float?
		@type param2: 				float?
		@type minRadius: 			int
		@type maxRadius: 			int
		@type windowSize:			int
		@type outlierThresholds:	list<float>?
		@type filterPupilSize:		bool
		@type surfaceBlur:			int?
		@type nThreads:				int?
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
		if param1 is not None:
			self.pupilFinder.param1 = param1
		if param2 is not None:
			self.pupilFinder.param2 = param2
		self.pupilFinder._minRadius = minRadius
		self.pupilFinder._maxRadius = maxRadius
		self.pupilFinder.FindPupils(bilateral = surfaceBlur, nThreads = nThreads)
		self.pupilFinder.FilterPupils(windowSize, outlierThresholds, filterPupilSize)


	def EstimateCalibrationPointPositions(self, beginTime = None, method = numpy.nanmedian, startDelay = 1.0 / 6.0, endSkip = 0.0, duration = None,
										  filtered = True, verbose = True):
		"""
		Estimates the pupil positions corresponding to each calibration point
		@param beginTime:		time of calibration sequence onset	TODO: replace this with a frame number
		@param method:			method used to aggregated points to one summary point
		@param startDelay:		time delay in seconds to account for eye movement delay
		@param endSkip:			time in seconds to clip from the end of a fixation period
		@param duration:		duration of each fixation point, if list, search through all and select best
		@param filtered:		use filtered pupil locations?
		@param verbose:			verbose?
		@type method:			function
		@type startDelay:		float
		@type endSkip:			float
		@type duration:			float|list<float>?
		@type filtered:			bool
		@type verbose:			bool
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
		@param smoothnesses:		smoothnesses to try for interpolations
		@param methods:				methods to try
		@param varianceThreshold:	threshold of variance in the calibration positions to throw away
		@param glintVector:			use pupil-glint vector instead of just the pupil position?
		@param searchThreshold:		if the error is above this threshold, search over delays/durations. will not search if is 0		
		@type smoothnesses:			list<float>
		@type methods:				list<str>
		@type varianceThreshold:	float?
		@type glintVector:			bool
		@type searchThreshold:		float
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
		@param durations: 	list of durations to search over
		@param delays: 		list of delay lengths to search over
		@param verbose: 	verbose?
		@type durations: 	list<float>
		@type delays: 		list<float>
		@type verbose: 		bool
		@return:	best duration, delay, and error
		@rtype:		tuple<float, float, float>
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
		@param pupilFinder:				pupil finder object that has been run
		@param trace:					output pupil location traces from a PupilFinder
		@param videoFileName:			video file to use
		@param replaceNanWithPrevious:	replace nan values with the previous value?
		@type pupilFinder:				PupilFinder?
		@type trace:					[n x 3] array?
		@type videoFileName:			str?
		@type replaceNanWithPrevious:	bool
		@return: screen coordinates corresponding to pupil location
		@rtype: numpy.ndarray
		"""

		# grumble grumble function overloading...
		isNone = [pupilFinder is not None, trace is not None, videoFileName is not None]
		if (sum(isNone) < 1):
			raise(ValueError('No inputs given'))
		if (sum(isNone) > 1):
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
		@type fileName: str
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
		@type fileName: str
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
