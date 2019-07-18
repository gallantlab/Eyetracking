import numpy
import cv2
import os
from EyetrackingCalibrator import EyetrackingCalibrator
from PupilFinder import PupilFinder
from skimage.transform import warp, AffineTransform
from skimage.draw import circle_perimeter as DrawCircle
from skimage import io


class Eyetracker(object):
	"""
	Top-level class for doing things with the eyetracker
	"""

	def __init__(self, calibrationFileName = None, dataFileName = None, calibrator = None, dataPupilFinder = None,
				 calibrationStart = None, dataStart = None, eyeWindow = None, calibrationPositions = None,
				 calibrationOrder = None,
				 config = None):
		"""
		Constructor
		@param calibrationFileName:		str?, name of eyetracking video for calibration
		@param dataFileName: 			str?, name of eyetracking video for data
		@param calibrator: 				EyetrackingCalibrator?, pre-constructed calibration object
		@param dataPupilFinder: 		PupilFinder?, pre-constructed pupil finding object
		@param calibrationStart: 		tuple<int, int, int, int>?, (H, M, S, MS) timestamp of the start of the calibration sequence in video
		@param dataStart: 				tuple<int, int, int, int>?, (H, M, S, MS) timestamp of the start of the data in video
		@param eyeWindow: 				tuple<int, int, int, int>?, (Left, right, top, bottom) window in the video to look for the pupil
		@param calibrationPositions: 	[n x 2] array<float>?, list of calibration point positions
		@param calibrationOrder: 		list<int>?, order of presentation of the calibraiton points
		@param config:
		"""

		self.calibrationFileName = calibrationFileName
		self.dataFileName = dataFileName

		self.calibrationStart = calibrationStart if calibrationStart is not None else (0, 0, 0, 0)
		self.dataStart = dataStart if dataStart is not None else (0, 0, 0, 0)
		self.eyeWindow = eyeWindow if eyeWindow is not None else (0, 320, 0, 240)
		self.calibrationOrder = calibrationOrder
		self.calibrationPositions = calibrationPositions

		if (config is not None):
			self.InitFromConfig(config)
		else:
			if calibrator is not None:
				self.calibrator = calibrator
			else:
				self.calibrator = None

			if dataPupilFinder:
				self.dataPupilFinder = dataPupilFinder
			else:
				self.dataPupilFinder = None


	def InitFromConfig(self, config):
		"""
		initializes from a configuration file
		@param config:
		@return:
		"""
		# TODO: this


	def FindPupils(self, blur = 5, dp = 1, minDistance = 600, param1 = 80, param2 = 80, minRadius = 10, maxRadius = 0, surfaceBlur = False):
		"""
		Find pupils in videos
		@param blur:
		@param dp:
		@param minDistance:
		@param param1:
		@param param2:
		@param minRadius:
		@param maxRadius:
		@param surfaceBlur:
		@return:
		"""
		pass


	def CalibrateEyetracking(self, startTime = None, calibrationOrder = None, calibrationPositions = None, filterLength = 29, varianceThreshold = None):
		"""
		Calibrates eyetracking and fits splines
		@param startTime:
		@param calibrationOrder:
		@param calibrationPositions:
		@param filterLength:
		@param varianceThreshold:
		@return:
		"""
		pass


	def WriteVideoWithGazePosition(self, frames, fileName, firstFrame = None, outResolution = (1024, 768), flipColors = True, fps = 30):
		"""
		Writes out a video using the input frames and given eyetracking
		@param frames: 			[t x h x w x 3] array, video frames, assumed to be same fps as the eyetracking
		@param fileName:		str, name of output file
		@param firstFrame: 		int?, the frame number in the eyetracking that the first frame of the video corresponds to
		@param outResolution: 	tuple<int, in>, desired output resolution
		@param flipColors:		bool, flip frame color channels order?
		@param fps:				float, fps of the frames
		@return:
		"""
		nFrames = frames.shape[0]
		video = cv2.VideoWriter(fileName, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, outResolution)

		gazeLocation = self.calibrator.TransformToScreenCoordinates(trace = self.dataPupilFinder.GetTraces(fps = fps))

		if ((frames.shape[1] != outResolution[1]) or (frames.shape[2] != outResolution[0])):
			hScale = frames.shape[2] * 1.0 / outResolution[0] * 1.0
			vScale = frames.shape[1] * 1.0 / outResolution[1] * 1.0
			transformation = AffineTransform(scale = [hScale, vScale], )
		else: transformation = None

		if not firstFrame:
			firstFrame = self.dataPupilFinder.FindOnsetFrame(self.dataStart[0], self.dataStart[1], self.dataStart[2], self.dataStart[3])

		for frame in range(nFrames):
			if (frame + firstFrame > gazeLocation.shape[0]):
				break
			gazeX = int(gazeLocation[frame + firstFrame, 0])
			gazeY = int(gazeLocation[frame + firstFrame, 1])
			image = frames[frame, :, :, :3].copy()
			if transformation is not None:
				image = warp(image, transformation, output_shape = (outResolution[1], outResolution[0])) * 255
			image[(gazeY - 5):(gazeY + 5), (gazeX - 10):(gazeX + 10), :] = [0, 0, 255]
			image[(gazeY - 10):(gazeY + 10), (gazeX - 5):(gazeX + 5), :] = [0, 0, 255]
			if flipColors:	# images are read in as RGB, but videos are written as BGR
				image = image[:, :, ::-1]
			video.write(image.astype(numpy.uint8))
		video.release()


	def WriteFramesWithGazePosition(self, frames, folder, firstFrame = None, outResolution = (1024, 768), flipColors = False, fps = 30):
		"""
		Writes out a video using the input frames and given eyetracking
		@param frames: 			[t x h x w x 3] array, video frames, assumed to be same fps as the eyetracking
		@param folder:			str, name of folder into which to write frames
		@param firstFrame: 		int?, the frame number in the eyetracking that the first frame of the video corresponds to
		@param outResolution: 	tuple<int, in>, desired output resolution
		@param flipColors:		bool, flip frame color channels order?
		@param fps:				float, fps of the frames
		@return:
		"""
		nFrames = frames.shape[0]
		gazeLocation = self.calibrator.TransformToScreenCoordinates(trace = self.dataPupilFinder.GetTraces(fps = fps))

		if ((frames.shape[1] != outResolution[1]) or (frames.shape[2] != outResolution[0])):
			hScale = frames.shape[2] * 1.0 / outResolution[0] * 1.0
			vScale = frames.shape[1] * 1.0 / outResolution[1] * 1.0
			transformation = AffineTransform(scale = [hScale, vScale], )
		else: transformation = None

		if not firstFrame:
			firstFrame = self.dataPupilFinder.FindOnsetFrame(self.dataStart[0], self.dataStart[1], self.dataStart[2], self.dataStart[3])

		if not (os.path.exists(folder)):
			os.makedirs(folder)

		for frame in range(nFrames):
			if (frame + firstFrame > gazeLocation.shape[0]):
				break
			gazeX = int(gazeLocation[frame + firstFrame, 0])
			gazeY = int(gazeLocation[frame + firstFrame, 1])
			image = frames[frame, :, :, :3].copy()
			if transformation is not None:
				image = warp(image, transformation, output_shape = (outResolution[1], outResolution[0])) * 255
			image[(gazeY - 5):(gazeY + 5), (gazeX - 10):(gazeX + 10), :] = [0, 0, 255]
			image[(gazeY - 10):(gazeY + 10), (gazeX - 5):(gazeX + 5), :] = [0, 0, 255]
			if flipColors:	# images are read in as RGB, but videos are written as BGR
				image = image[:, :, ::-1]
			io.imsave(folder + '/frame-{:06d}.png'.format(frame), image)


	def RecenterFramesToVideo(self, frames, fileName, scale = 0.125, firstFrame = None, flipColors = True, padValue = 0.1, gazeSize = (1024, 768), fps = 30):
		"""
		Writes out a video in which the frames are moved such that the gaze position is always
		the center
		@param frames: 		[t x w x h x 3] array, video frames, assumed ot be same fps as eyetracking
		@param fileName: 	str, name of output file
		@param scale: 		float, scale of these frames relative to the eyetracking scale (typically 1024 x 768)
		@param firstFrame: 	int?, the frame number in the eyetracking that the first frame of the video corresponds to
		@param flipColors:	bool, flip frame color channel order?
		@param padValue:	float, range [0, 1] value to use for padding in the recentered frames
		@param gazeSize:	tuple<int, int>, stimuli resolution on to which eyetracking is mapepd
		@param fps:			float, fps of the frames
		@return:
		"""
		nFrames = frames.shape[0]
		width = frames.shape[2]
		height = frames.shape[1]
		video = cv2.VideoWriter(fileName, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (width * 2, height * 2))
		gazeLocation = numpy.nan_to_num(self.calibrator.TransformToScreenCoordinates(trace = self.dataPupilFinder.GetTraces()))

		if not firstFrame:
			firstFrame = self.dataPupilFinder.FindOnsetFrame(self.dataStart[0], self.dataStart[1], self.dataStart[2], self.dataStart[3])

		# basically, use stimulus frame time as the standard, and convert/index into the gazeLocation vector with that
		frameIndexingFactor = self.dataPupilFinder.fps / fps

		for frame in range(nFrames):
			eyetrackingFrame = int(frame * frameIndexingFactor) + firstFrame
			if (eyetrackingFrame > gazeLocation.shape[0]):
				break
			dx = int(gazeSize[0] / 2 - gazeLocation[eyetrackingFrame, 0])
			dy = int(gazeSize[1] / 2 - gazeLocation[eyetrackingFrame, 1])
			transform = AffineTransform(translation = [dx * scale + width / 2, dy * scale + height / 2])
			image = warp(frames[frame, :, :, :3], transform.inverse, output_shape = (height * 2, width * 2), cval = padValue) * 255
			image = image.astype(numpy.uint8)
			if flipColors:
				image = image[:, :, ::-1]
			video.write(image)
		video.release()


	def RecenterFrames(self, frames, folder, scale = 0.125, firstFrame = None, flipColors = False, padValue = 0.1, gazeSize = (1024, 768), fps = 30):
		"""
		Writes recentered frames to a folder
		@param frames: 		[t x w x h x 3] array, video frames, assumed ot be same fps as eyetracking
		@param folder: 		str, name of folder into which to write frames
		@param scale: 		float, scale of these frames relative to the eyetracking scale (typically 1024 x 768)
		@param firstFrame: 	int?, the frame number in the eyetracking that the first frame of the video corresponds to
		@param flipColors:	bool, flip frame color channel order?
		@param padValue:	float, range [0, 1] value to use for padding
		@param gazeSize:	tuple<int, int>, stimuli resolution on to which eyetracking is mapepd
		@param fps:			float, fps of the frames
		@return:
		"""
		nFrames = frames.shape[0]
		width = frames.shape[2]
		height = frames.shape[1]
		gazeLocation = numpy.nan_to_num(self.calibrator.TransformToScreenCoordinates(trace = self.dataPupilFinder.GetTraces()))

		if not firstFrame:
			firstFrame = self.dataPupilFinder.FindOnsetFrame(self.dataStart[0], self.dataStart[1], self.dataStart[2], self.dataStart[3])

		# basically, use stimulus frame time as the standard, and convert/index into the gazeLocation vector with that
		frameIndexingFactor = self.dataPupilFinder.fps / fps

		if not (os.path.exists(folder)):
			os.makedirs(folder)

		for frame in range(nFrames):
			eyetrackingFrame = int(frame * frameIndexingFactor) + firstFrame
			if (eyetrackingFrame > gazeLocation.shape[0]):
				break
			dx = int(gazeSize[0] / 2 - gazeLocation[eyetrackingFrame, 0])
			dy = int(gazeSize[1] / 2 - gazeLocation[eyetrackingFrame, 1])
			transform = AffineTransform(translation = [dx * scale + width / 2, dy * scale + height / 2])
			image = warp(frames[frame, :, :, :3], transform.inverse, output_shape = (height * 2, width * 2), cval = padValue) * 255
			image = image.astype(numpy.uint8)
			if flipColors:
				image = image[:, :, ::-1]
			io.imsave(folder + '/frame-{:06d}.png'.format(frame), image)


	def WriteSideBySideVideo(self, frames, fileName, firstDataFrame = None, firstEyetrackingFrame = None, stimuliResolution = (1024, 768), flipColors = True, stimulusFPS = 30):
		"""
		Writes a video out with the eyetracking video and stimulus with gaze position side by side
		@param frames:					[time x h x w x 3] stimulus frame array
		@param fileName:				str, name for video file to write
		@param firstDataFrame:
		@param firstEyetrackingFrame:
		@param stimuliResolution:
		@param flipColors:				bool, reverse the colors dimension of the frames when writing the video?
		@param stimulusFPS:				int, fps of the frames
		@return:
		"""
		nFrames = frames.shape[0]
		gazeLocation = self.calibrator.TransformToScreenCoordinates(trace = self.dataPupilFinder.filteredPupilLocations)

		if ((frames.shape[1] != stimuliResolution[1]) or (frames.shape[2] != stimuliResolution[0])):
			hScale = frames.shape[2] * 1.0 / stimuliResolution[0] * 1.0
			vScale = frames.shape[1] * 1.0 / stimuliResolution[1] * 1.0
			transformation = AffineTransform(scale = [hScale, vScale], )
		else: transformation = None

		if not firstDataFrame:
			firstDataFrame = self.dataPupilFinder.FindOnsetFrame(self.dataStart[0], self.dataStart[1], self.dataStart[2], self.dataStart[3])

		if not firstEyetrackingFrame:
			firstEyetrackingFrame = self.calibrator.pupilFinder.FindOnsetFrame(self.calibrationStart[0], self.calibrationStart[1], self.calibrationStart[2], self.calibrationStart[3])

		eyetrackingFramesPerStimulusFrame = int(self.dataPupilFinder.fps / stimulusFPS)

		circles = self.dataPupilFinder.filteredPupilLocations.astype(numpy.int)
		video = cv2.VideoWriter(fileName, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), self.dataPupilFinder.fps, (stimuliResolution[0] + 320, stimuliResolution[1] if stimuliResolution[1] > 240 else 240))
		thisFrame = numpy.zeros([stimuliResolution[1] if stimuliResolution[1] > 240 else 240, stimuliResolution[0] + 320, 3], dtype = numpy.uint8)
		for frame in range(nFrames * eyetrackingFramesPerStimulusFrame):	# frame is output video frame, locked to pupil video fps
			dataFrame = frame + firstDataFrame								# indexes correctly into the eyetracking video
			image = frames[frame / eyetrackingFramesPerStimulusFrame, :, :, :3].copy()
			if transformation is not None:
				image = warp(image, transformation, output_shape = (stimuliResolution[1], stimuliResolution[0])) * 255
			x = int(gazeLocation[dataFrame, 0])
			y = int(gazeLocation[dataFrame, 1])
			image[(y - 5):(y + 5), (x - 10):(x + 10), :] = [0, 0, 255]
			image[(y - 10):(y + 10), (x - 5):(x + 5), :] = [0, 0, 255]
			thisFrame[:, 320:, :] = image
			thisFrame[:240, :320, :] = self.dataPupilFinder.rawFrames[dataFrame, :, :, ::-1].copy()
			if not self.dataPupilFinder.filteredPupilLocations[dataFrame, 0] == numpy.nan:
				if (not self.dataPupilFinder.blinks[dataFrame]) and (not numpy.any(numpy.isnan(self.dataPupilFinder.filteredPupilLocations[dataFrame, :]))):
					for radiusOffset in range(-2, 3):
						y, x = DrawCircle(circles[dataFrame, 0], circles[dataFrame, 1], circles[dataFrame, 2] + radiusOffset, shape = (240, 320))
						thisFrame[x, y, 2] = 255
						thisFrame[(circles[dataFrame, 1] - 4):(circles[dataFrame, 1] + 4), (circles[dataFrame, 0] - 1):(circles[dataFrame, 0] + 1), 2] = 255
						thisFrame[(circles[dataFrame, 1] - 1):(circles[dataFrame, 1] + 1), (circles[dataFrame, 0] - 4):(circles[dataFrame, 0] + 4), 2] = 255
			if flipColors:
				thisFrame = thisFrame[:, :, ::-1]
			video.write(thisFrame.astype(numpy.uint8))
		video.release()