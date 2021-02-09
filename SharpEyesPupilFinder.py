from .PupilFinder import PupilFinder
import numpy

class DummyException(Exception):
	"""
	Raised when an unused method is called on a dummy class object
	"""
	pass


class SharpEyesPupilFinder(PupilFinder):
	"""
	Dummy class that takes in SharpEyes outputs. Yes, the dummy exceptions are basically
	and anti-pattern, but I'm too lazy to replicate the pupil filtering capabilities.
	"""
	def __init__(self, pupilPositionsFile, timeStampsFile, fps, confidenceThreshold = 0.6):
		"""
		Constructor
		@param pupilPositionsFile:	position of pupils as found by SharpEyes
		@param timeStampsFile: 		timestamp of each frame as found by SharpEyes
		@param fps:					fps of the video parsed
		@param confidenceThreshold:	value at which to say that it's a blink
		"""
		self.rawPupilLocations = numpy.load(pupilPositionsFile)
		self.time = numpy.load(timeStampsFile)
		self.isParsed = True
		self.fps = fps
		self.filteredPupilLocations = None
		self.nFrames = self.rawPupilLocations.shape[0]
		self.blinks = self.rawPupilLocations[:, 3] < confidenceThreshold
		self.hasGlint = False

	def InitFromOther(self, other):
		raise DummyException

	def GetTimeStampForFrame(self, frameIndex):
		raise DummyException

	def ParseTimestamps(self, nThreads = 1):
		raise DummyException

	def FindPupils(self, endFrame = None, bilateral = None, nThreads = 0):
		raise DummyException

	def WritePupilFrames(self, directory, startFrame = None, endFrame = None, filtered = True, burnLocation = True):
		raise DummyException

	def WritePupilVideo(self, fileName, startFrame = None, endFrame = None, filtered = True, burnLocation = True):
		raise DummyException

	def WriteVideo(self, outFileName, frames = None):
		raise DummyException