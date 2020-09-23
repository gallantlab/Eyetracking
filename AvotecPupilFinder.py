import numpy

from .PupilFinder import  PupilFinder


class AvotecPupilFinder(PupilFinder):
	"""
	Parses the Avotec output files for values found by avotec.
	Comes in useful when we don't have raw videos or have troubles with the raw video files
	This class inherits from PupilFinder only because we want to keep the inheritance structure
	and the interface provided by PupilFinder so we can just drop this class in place where
	the video-based pupil finders are used.

	In this case, the filtered pupil positions are the avotec corrected gaze location
	"""

	def __init__(self, dataFileName):
		"""
		Constructor
		@param dataFileName:	name of text file output my Avotec
		@type dataFileName:		str
		"""
		self.dataFileName = dataFileName
		"""
		@ivar:	name of the data file
		@type:	str
		"""

		self.quality = numpy.array([])
		"""
		@ivar:	quality at each frame as determined by avotec
		@type:	numpy.ndarray
		"""

		self.pupilSize = numpy.array([])
		"""
		@ivar:	width and height of the pupil ellipse
		@type:	numpy.ndarray
		"""

		self.isParsed = True
		self.fps = 0
		self.FindPupils()


	def FindPupils(self, endFrame = None, bilateral = None, nThreads = 0):
		"""
		Parses the avotec file
		@param endFrame: 	useless override to keep the same signature
		@param bilateral: 	useless override to keep the same signature
		@param nThreads: 	useless override to keep the same signature
		@return:
		"""
		dataFile = open(self.dataFileName)
		time = []			# time in seconds
		rawGaze = []		# raw x, y gaze pupil size (average of width and height)
		quality = []		# quality at each frame as determined by avotec
		correctedGaze = []	# avotec corrected gaze
		pupilSize = []		# width, height of pupil

		line = dataFile.readline()
		while line != '':
			tokens = line.split('\t')
			if tokens[0] == '7':
				self.fps = float(tokens[3])
			elif tokens[0] == '10':	# 10 indicates that this is a data entry
				values = [float(item) for item in tokens[:13]]
				rawGaze.append((values[3], values[4], (values[7] + values[8]) / 2.0))
				quality.append(values[10])
				correctedGaze.append((values[5], values[6]))
				pupilSize.append((values[7], values[8]))

				t = values[1]
				hours = int(t / 3600)
				t -= hours * 3600
				minutes = int(t / 60)
				t -= minutes * 60
				seconds = int(t)
				milliseconds = int(numpy.round((t - seconds) * 1000))
				time.append((hours, minutes, seconds, milliseconds))
			line = dataFile.readline()

		self.rawPupilLocations = numpy.array(rawGaze)
		self.quality = numpy.array(quality)
		self.pupilSize = numpy.array(pupilSize)
		self.filteredPupilLocations = numpy.array(correctedGaze)
		self.time = numpy.array(time, dtype = numpy.int)
		self.nFrames = self.time.shape[0]


	def WritePupilFrames(self, directory, startFrame = None, endFrame = None, filtered = True, burnLocation = True):
		"""
		Override to disable
		@param directory:
		@param startFrame:
		@param endFrame:
		@param filtered:
		@param burnLocation:
		@return:
		"""
		print('This object uses the Avotec file and does not have frames to write')
		return


	def WritePupilVideo(self, fileName, startFrame = None, endFrame = None, filtered = True, burnLocation = True):
		"""
		Override to disable
		@param fileName:
		@param startFrame:
		@param endFrame:
		@param filtered:
		@param burnLocation:
		@return:
		"""
		print('This object uses the Avotec file and does not have frames to write')
		return


	def ParseTimestamps(self, nThreads = 1):
		"""
		Empty override
		@param nThreads:
		@return:
		"""
		return


	def FilterPupils(self, windowSize = 15, outlierThresholds = None, filterPupilSize = True):
		"""
		Empty override because the filtered locations are the avotec filtering
		@param windowSize:
		@param outlierThresholds:
		@param filterPupilSize:
		@return:
		"""
		return