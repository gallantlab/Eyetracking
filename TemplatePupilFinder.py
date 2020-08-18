import numpy
import threading
import cv2
from zipfile import ZipFile
from .EyetrackingUtilities import SaveNPY, ReadNPY
from .PupilFinder import PupilFinder, median2way, outliers2nan
from skimage.draw import circle, circle_perimeter as DrawCircle

class TemplatePupilFinder(PupilFinder):
	"""
	A class that finds pupils by template matching and not fitting a circle.
	A pupil is just a black circle on a white background
	A glint is just a white circle on a black background close to the pupil
	"""
	@staticmethod
	def Worker(rawFrames, radii, window, bilateral, blur, rawPupilLocations, rawGlintLocations,
			   pupilTemplates, glintTemplates):
		"""
		Actual code for template-matching, factored out to a method so it could be multithreaded
		@param rawFrames:			[frame x width x height x3] frame array
		@param radii:				list<int> radii to be used to for template matching
		@param window:				tuple<int, int, int, int>? window in frames to use
		@param bilateral:			int?, bilateral fiter size
		@param blur:				int, mediam flur size
		@param rawPupilLocations:	[frame x value] where to store found pupil locations
		@param rawGlintLocations:	[frame x value] where to store found glint locations
		@param pupilTemplates:		[w x h x num] templates for pupils
		@param glintTemplates:		[w x h x num] templates for glints
		@return:
		"""
		thesePupilPositions = numpy.zeros([len(radii), 2])
		thesePupilCorrelations = numpy.zeros(len(radii))
		theseGlintPositions = numpy.zeros([9, 2])
		theseGlintCorrelations = numpy.zeros([9])

		for frameIndex in range(rawFrames.shape[0]):
			# === find pupil ===
			if window is not None:
				frame = rawFrames[frameIndex, window[2]:window[3], window[0]:window[1], :].mean(-1).astype(numpy.uint8)
			else:
				frame = rawFrames[frameIndex, :, :, :].mean(-1).astype(numpy.uint8)
			if (bilateral is not None) and (bilateral > 0):
				frame = cv2.bilateralFilter(frame, bilateral, 100, 75)
			if (blur > 0):
				frame = cv2.medianBlur(frame, blur)
			for i in range(len(radii)):
				res = cv2.matchTemplate(frame, pupilTemplates[:, :, i], cv2.TM_CCOEFF_NORMED)
				_, maxCorr, _, maxPos = cv2.minMaxLoc(res)
				thesePupilPositions[i, :] = maxPos
				thesePupilCorrelations[i] = maxCorr
			best = numpy.argmax(thesePupilCorrelations)
			rawPupilLocations[frameIndex, 3] = thesePupilCorrelations[best]
			rawPupilLocations[frameIndex, :2] = thesePupilPositions[best, :] + 25
			if window is not None:
				rawPupilLocations[frameIndex, 0] += window[0]
				rawPupilLocations[frameIndex, 1] += window[2]
			rawPupilLocations[frameIndex, 2] = radii[best]

			# === find glint ===
			x = int(thesePupilPositions[best, 0] + 25)
			y = int(thesePupilPositions[best, 1] + 25)
			frame = frame[(y - 20):(y + 20), (x - 20):(x + 20)]  # glint is in close vicinity of the pupil
			for i in range(9):
				res = cv2.matchTemplate(frame, glintTemplates[:, :, i], cv2.TM_CCOEFF_NORMED)
				_, maxCorr, _, maxPos = cv2.minMaxLoc(res)
				theseGlintPositions[i, :] = maxPos
				theseGlintCorrelations[i] = maxCorr
			best = numpy.argmax(theseGlintCorrelations)
			rawGlintLocations[frameIndex, 3] = theseGlintCorrelations[best]
			rawGlintLocations[frameIndex, :2] = theseGlintPositions[best, :] + 7
			rawGlintLocations[frameIndex, 0] += (x - 20)
			rawGlintLocations[frameIndex, 1] += (y - 20)
			if window is not None:
				rawGlintLocations[frameIndex, 0] += window[0]
				rawGlintLocations[frameIndex, 1] += window[2]
			rawGlintLocations[frameIndex, 2] = (best + 2) / 2.0


	def __init__(self, videoFileName = None, window = None, minRadius = 13, maxRadius = 23, other = None):
		"""
		Constructor
		@param videoFileName:	str?, name of video file to parse
		@param window: 			4-ple<int>?, subwindow in frame to examine, order [left, right, top bottom]
		@param minRadius:		int, smallest radius to look for
		@param maxRadius:		int, biggest radius to look for, should not be bigger than 25
		@param other:			VideoReader?, object to copy construct from
		"""
		super(TemplatePupilFinder, self).__init__(videoFileName, window, other = other)
		self.radii = range(minRadius, maxRadius + 1)
		self.pupilTemplates = numpy.ones([51, 51, len(self.radii)], numpy.uint8) * 255
		self.glintTemplates = numpy.zeros([15, 15, 9], numpy.uint8)	# glint isn't that big, do radius range 1-5 px
		for i in range(len(self.radii)):
			y, x = circle(25, 25, self.radii[i])
			self.pupilTemplates[x, y, i] = 0
		for i in range(2, 11):
			y, x = circle(7, 7, i / 2.0)
			self.glintTemplates[x, y, i - 2] = 255

		self.rawGlintLocations = None
		self.filteredGlintLocations = None


	def InitFromOther(self, other):
		"""
		Jank copy constructor
		@param other: 	TemplatPupilFinder
		@return:
		"""
		super(TemplatePupilFinder, self).InitFromOther(other)
		if other.rawGlintLocations is not None:
			self.rawGlintLocations = other.rawGlintLocations.copy()
		if other.filteredGlintLocations is not None:
			self.filteredGlintLocations = other.filteredGlintLocations.copy()


	def FindPupils(self, endFrame = None, bilateral = None, nThreads = 1):
		"""
		Finds pupils by template matching
		@param endFrame:	int?, frame to search to
		@param bilateral:	bool, useless here, but is overridden from super function
		@param nThreads:		int, number of threads to use
		@return:
		"""
		if ((endFrame is None) or endFrame > self.nFrames):
			endFrame = self.nFrames

		# self.frameDiffs = numpy.r_[0, numpy.sum(numpy.diff(self.rawFrames, axis = 0) ** 2, (1, 2, 3))]
		# self.blinks = numpy.where(self.frameDiffs > self.frameDiffs.mean() + self.frameDiffs.std() * 2, True, False)

		self.rawPupilLocations = numpy.zeros([endFrame, 4])		# here, the colums are [x, y, radius, confidence]
		self.rawGlintLocations = numpy.zeros([endFrame, 4])
		# === parallel for ===
		if nThreads == 1:
			TemplatePupilFinder.Worker(self.rawFrames[:endFrame, :, :, :], self.radii, self.window,
									   bilateral, self.blur, self.rawPupilLocations, self.rawGlintLocations,
									   self.pupilTemplates, self.glintTemplates)

		else:
			chunkSize = int(endFrame / nThreads)
			threads = []
			for thread in range(nThreads):
				start = chunkSize * thread
				end = start + chunkSize
				if thread == (nThreads - 1):
					end = endFrame
				threads.append(threading.Thread(target = TemplatePupilFinder.Worker,
												args = (self.rawFrames[start:end, :, :, :], self.radii, self.window,
														bilateral, self.blur, self.rawPupilLocations[start:end, :],
														self.rawGlintLocations[start:end, :], self.pupilTemplates,
														self.glintTemplates)))
			for thread in threads:
				thread.start()
			for thread in threads:
				thread.join()
		self.blinks = numpy.where(self.rawPupilLocations[:, 3] < (numpy.mean(self.rawPupilLocations[:, 3]) - 1.5 * numpy.std(self.rawPupilLocations[:, 3])), True, False)	# less than -1.5 std confidence = blink


	def FilterPupils(self, windowSize = 15, outlierThresholds = None, filterPupilSize = True):
		"""
		Filters raw pupil and glint locations
		@param windowSize:			int, median filter time window size
		@param outlierThresholds:	list<float>?, thresholds in percentiles at which to nan outliers, if none, does not nan outliers
		@param filterPupilSize:		bool, filter pupil size alone with position?
		@return:
		"""
		super(TemplatePupilFinder, self).FilterPupils(windowSize, outlierThresholds, filterPupilSize)

		self.filteredGlintLocations = self.rawGlintLocations.copy()
		for i in range(3 if filterPupilSize else 2):
			self.filteredGlintLocations[:, i] = median2way(self.filteredGlintLocations[:, i], windowSize)

		for i in range(self.nFrames):
			if self.blinks[i]:
				self.filteredGlintLocations[i, :] = numpy.nan

		if (outlierThresholds is not None):
			for i in range(3):
				if (outlierThresholds[i] is not None):
					self.filteredGlintLocations[:, i] = outliers2nan(self.filteredGlintLocations[:, i], outlierThresholds[i])
			# 1 nan in row => entire row nan
			self.filteredGlintLocations[numpy.isnan(self.filteredGlintLocations.sum(axis = -1))] = numpy.nan


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

		pupils = self.filteredPupilLocations.astype(numpy.int) if filtered else self.rawPupilLocations.astype(numpy.int)
		glints = self.filteredGlintLocations.astype(numpy.int) if filtered else self.rawGlintLocations.astype(numpy.int)

		video = cv2.VideoWriter(fileName, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), self.fps, (self.width, self.height))
		image = numpy.zeros_like(self.rawFrames[0, :, :, :])
		for frame in range(startFrame, endFrame):
			image = self.rawFrames[frame, :, :, :].copy()
			# image[:, :, 1] = image[:, :, 0]
			# image[:, :, 2] = image[:, :, 0]
			if not (filtered and self.filteredPupilLocations[frame, 0] == numpy.nan):
				if (not self.blinks[frame]) and (not numpy.any(numpy.isnan(self.filteredPupilLocations[frame, :]))):
					# outline pupil
					for radiusOffset in range(-2, 3):
						y, x = DrawCircle(pupils[frame, 0], pupils[frame, 1], pupils[frame, 2] + radiusOffset, shape = (self.height, self.width))
						image[x, y, 2] = 255
					image[(pupils[frame, 1] - 4):(pupils[frame, 1] + 4), (pupils[frame, 0] - 1):(pupils[frame, 0] + 1), 2] = 255
					image[(pupils[frame, 1] - 1):(pupils[frame, 1] + 1), (pupils[frame, 0] - 4):(pupils[frame, 0] + 4), 2] = 255
					# outline glint
					for radiusOffset in range(0, 2):
						y, x = DrawCircle(glints[frame, 0], glints[frame, 1], glints[frame, 2] + radiusOffset, shape = (self.height, self.width))
						image[x, y, 0] = 255
					if burnLocation:
						cv2.putText(image, 'x: {:03d} y: {:03d} r: {:03d}'.format(pupils[frame, 0], pupils[frame, 1], pupils[frame, 2]), (10, 20), cv2.FONT_HERSHEY_DUPLEX, 0.75, [0, 255, 0])
						cv2.putText(image, 'x: {:03d} y: {:03d} r: {:03d}'.format(glints[frame, 0], glints[frame, 1], glints[frame, 2]), (10, 45), cv2.FONT_HERSHEY_DUPLEX, 0.75, [0, 255, 0])
				else:
					if burnLocation:
						cv2.putText(image, 'Blink', (10, 20), cv2.FONT_HERSHEY_DUPLEX, 0.75, [0, 255, 0])
						cv2.putText(image, 'Blink', (10, 45), cv2.FONT_HERSHEY_DUPLEX, 0.75, [0, 255, 0])
				if burnLocation:
					cv2.putText(image, 'frame {:06d}'.format(frame), (10, 70), cv2.FONT_HERSHEY_DUPLEX, 0.75, [0, 255, 0])
			video.write(image)
		video.release()

	def Save(self, fileName = None, outFile = None):
		"""
		Save out information
		@param fileName: 	str?, name of file to save, must be not none if fileObject is None
		@param outFile: 	zipfile?, existing object to write to
		@return:
		"""

		closeOnFinish = outFile is None  # we close the file only if this is the actual function that started the file

		if outFile is None:
			outFile = ZipFile(fileName, 'w')

		super(TemplatePupilFinder, self).Save(None, outFile)

		if self.rawGlintLocations is not None:
			SaveNPY(self.rawGlintLocations, outFile, 'rawGlintLocations.npy')
		if self.filteredGlintLocations is not None:
			SaveNPY(self.filteredGlintLocations, outFile, 'filteredGlintLocations.npy')

		if closeOnFinish:
			outFile.close()

	def Load(self, fileName = None, inFile = None):
		"""
		Loads in information
		@param fileName: 	str? name of file to read, must not be none if infile is none
		@param inFile:		zipfile? existing object to read from
		@return:
		"""

		closeOnFinish = inFile is None
		if inFile is None:
			inFile = ZipFile(fileName, 'r')

		super(TemplatePupilFinder, self).Load(None, inFile)

		subFiles = inFile.NameToInfo.keys()
		if 'rawPupilLocations.npy' in subFiles:
			self.rawPupilLocations= ReadNPY(inFile, 'rawPupilLocations.npy')
		if 'frameDiffs.npy' in subFiles:
			self.frameDiffs = ReadNPY(inFile, 'frameDiffs.npy')
		if 'blinks.npy' in subFiles:
			self.blinks = ReadNPY(inFile, 'blinks.npy')
		if 'filteredPupilLocations.npy' in subFiles:
			self.filteredPupilLocations = ReadNPY(inFile, 'filteredPupilLocations.npy')
		if 'filteredGlintLocations.npy' in subFiles:
			self.filteredGlintLocations = ReadNPY(inFile, 'filteredGlintLocations.npy')
		if 'rawGlintLocations.npy' in subFiles:
			self.rawGlintLocations = ReadNPY(inFile, 'rawGlintLocations.npy')

		if closeOnFinish:
			inFile.close()