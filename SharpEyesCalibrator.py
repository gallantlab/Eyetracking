from . import EyetrackingCalibrator
from .SharpEyesPupilFinder import SharpEyesPupilFinder

class SharpEyesCalibrator(EyetrackingCalibrator.EyetrackingCalibrator):
	"""
	Calibrator class that makes use of the SharpEyes outputs.
	Subclassed from EyetrackingCalibrator to hide the video reading stuff
	"""

	def __init__(self, pupilPositionsFile, timeStampsFile, calibrationBeginTime = None, calibrationPositions = None,
				 calibrationOrder = None, calibrationDuration = 2, calibrationDelay = 2, fps = 60):
		"""
		Constructor, takes positions and timestamps as found by SharpEyes
		@param pupilPositionsFile:		position of pupils found by SharpEyes (2D array)
		@type pupilPositionsFile:		str, name of saved numpy array
		@param timeStampsFile:			timestamps of each frame found by SharpEyes
		@type timeStampsFile:			str, name of saved numpy array
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
		@param fps:						fps of the video
		@type fps:						double
		"""
		super(SharpEyesCalibrator, self). __init__(None, calibrationBeginTime, calibrationPositions,
												  calibrationOrder, calibrationDuration, calibrationDelay)
		self.pupilPositionsFile = pupilPositionsFile
		self.timestampsFile = timeStampsFile
		self.fps = fps
		self.hasGlint = False


	def FindPupils(self, blinkConfidence = 0.6, window = None, blur = 5, dp = 1, minDistance = 600, param1 = None,
				   param2 = None, minRadius = 10, maxRadius = 18, windowSize = 15, outlierThresholds = None,
				   filterPupilSize = True, surfaceBlur = None, nThreads = None):
		"""
		Builds the dummy pupil finder object
		"""
		self.pupilFinder = SharpEyesPupilFinder(self.pupilPositionsFile, self.timestampsFile, self.fps, blinkConfidence)