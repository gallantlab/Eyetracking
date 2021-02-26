import numpy
import cv2
import os
from .EyetrackingCalibrator import EyetrackingCalibrator
from skimage.transform import warp, AffineTransform
from skimage.draw import circle_perimeter as DrawCircle
from skimage import io


class Eyetracker(object):
	"""
	Top-level class for doing things with the eyetracker
	"""

	def __init__(self, calibrator, dataPupilFinder = None):
		"""
		Constructor
		@param calibrator: 				pre-constructed calibration object that has been fit
		@param dataPupilFinder: 		pre-constructed pupil finding object and pupils have been found, if none, will be that from the calibrator
		@type calibrator: 				EyetrackingCalibrator
		@type dataPupilFinder: 			PupilFinder?
		"""
		self.calibrator = calibrator
		"""
		@ivar: Object used to compute the pupil to gaze mapping on the calibration sequence
		@type: EyetrackingCalibrator
		"""

		if dataPupilFinder is not None:
			self.dataPupilFinder = dataPupilFinder
			"""
			@ivar: Object used to find pupil position in the actual data
			@type: PupilFinder
			"""
		else:
			self.dataPupilFinder = calibrator.pupilFinder


	def GetGazeLocations(self, filtered = True):
		"""
		Gets the gaze positions from the data
		@param filtered:	use temporally filtered traces?
		@type filtered:		bool
		@return: estimated gaze locations in screen space
		@rtype:	numpy.ndarray
		"""
		return self.calibrator.TransformToScreenCoordinates(self.dataPupilFinder.GetTraces(filtered))


	def WriteVideosWithGazePosition(self, videoFileName, outFileName = None, eyetrackingStartTime = None, firstEyetrackingFrame = None,
									firstFrameInVideo = None, flipColors = False):
		"""
		Reads in a video, and writes the eyetracking position on the frames
		@param videoFileName: 			video frame to read in
		@param outFileName: 			video file name to write out
		@param eyetrackingStartTime: 	timestamp of the start of the data frames in eyetrakcing, has precedence over firstEyetrackingFrame
		@param firstEyetrackingFrame: 	the frame in the eyetracking that the first frame of the video corresponds to
		@param firstFrameInVideo: 		the frame in the video at which the data start, if None, will use the TTL marker/gray screen to figure it out
		@note firstFrameInVideo is specific to the driving project
		@param flipColors: 				flip colors because BGR->RGB?
		@return:
		"""
		if (eyetrackingStartTime is None) and (firstEyetrackingFrame is None):
			dataStart = self.calibrator.calibrationBeginTime
		if (eyetrackingStartTime is not None) and (firstEyetrackingFrame is not None):
			print('Both dataStart and firstFrame are provided, using dataStart')

		videoIn = cv2.VideoCapture(videoFileName)
		if (not videoIn.isOpened()):
			raise ValueError('Could not open video file {}'.format(videoFileName))

		if (outFileName is None):
			tokens = os.path.splitext(videoFileName)
			outFileName = tokens[0] + ' with gaze position.avi'

		width = int(videoIn.get(cv2.CAP_PROP_FRAME_WIDTH))
		height = int(videoIn.get(cv2.CAP_PROP_FRAME_HEIGHT))
		fps = videoIn.get(cv2.CAP_PROP_FPS)

		videoOut = cv2.videoWriter(outFileName, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (width, height))

		gazeLocation = self.calibrator.TransformToScreenCoordinates(trace = self.dataPupilFinder.GetTraces(fps = fps))

		success = True
		if (firstFrameInVideo is not None):
			for i in range(firstFrameInVideo):
				videoIn.read()
		else:
			print('Warning: if this is not a video from the driving project, results will be garbage')

			nFramesRead = 0
			TTLTemplate = numpy.load(os.path.dirname(__file__) + '/indicator-template.npy').ravel()

			while (success):
				nFramesRead += 1
				success, frame = videoIn.read()
				patch = frame[745:763, 1000:1018, :] > 196  # lower right corner in which TTL indicator appears
				patch = patch.mean(2)
				if (numpy.corrcoef(TTLTemplate, patch.ravel())[0, 1] > 0.9):
					break
				elif (numpy.sum(frame[745:763, 1000:1018, :] - 137) == 0):
					nFramesRead -= 2
					break

			videoIn.set(cv2.CAP_PROP_POS_FRAMES, nFramesRead)

		color = [30, 144, 255] if not flipColors else [255, 144, 30]

		dataFrame = 0
		while (success):
			success, frame = videoIn.read()
			gazeX = int(gazeLocation[dataFrame + firstEyetrackingFrame, 0])
			gazeY = int(gazeLocation[dataFrame + firstEyetrackingFrame, 1])

			frame[(gazeY - 5):(gazeY + 5), (gazeX - 10):(gazeX + 10), :] = color
			frame[(gazeY - 10):(gazeY + 10), (gazeX - 5):(gazeX + 5), :] = color

			videoOut.write(frame)

		videoIn.release()
		videoOut.release()


	def WriteVideoWithGazePositionFromFrames(self, frames, fileName, dataStart = None, firstFrame = None,
								   			 outResolution = (1024, 768), flipColors = True, fps = 30):
		"""
		Writes out a video using the input frames and given eyetracking.
		If neither dataStart nor firstFrame are provided, then the assumption is that eyetracking calibration
		is included in the data, and that the data start is the eyetracking calibration start.
		@param frames: 			assumed to be same fps as the eyetracking
		@param fileName:		name of output file
		@param dataStart:		timestamp of the start of the data frames, has precedence over firstFrame
		@param firstFrame: 		the frame number in the eyetracking that the first frame of the video corresponds to
		@param outResolution: 	desired output resolution
		@param flipColors:		flip frame color channels order? original videos are in BGR
		@param fps:				float, fps of the frames
		@type frames: 			[t x h x w x 3] numpy.ndarray
		@type fileName:			str
		@type dataStart:		tuple<int, int, int, int>
		@type firstFrame: 		int?
		@type outResolution: 	tuple<int, int>
		@type flipColors:		bool
		@type fps:				float
		"""
		if (dataStart is None) and (firstFrame is None):
			dataStart = self.calibrator.calibrationBeginTime
		if (dataStart is not None) and (firstFrame is not None):
			print('Both dataStart and firstFrame are provided, using dataStart')

		nFrames = frames.shape[0]
		video = cv2.VideoWriter(fileName, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, outResolution)

		gazeLocation = self.calibrator.TransformToScreenCoordinates(trace = self.dataPupilFinder.GetTraces(fps = fps))

		if ((frames.shape[1] != outResolution[1]) or (frames.shape[2] != outResolution[0])):
			hScale = frames.shape[2] * 1.0 / outResolution[0] * 1.0
			vScale = frames.shape[1] * 1.0 / outResolution[1] * 1.0
			transformation = AffineTransform(scale = [hScale, vScale], )
		else: transformation = None

		if dataStart is not None:
			firstFrame = self.dataPupilFinder.FindOnsetFrame(dataStart[0], dataStart[1], dataStart[2], dataStart[3])

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


	def WriteFramesWithGazePosition(self, frames, folder, dataStart = None, firstFrame = None,
									outResolution = (1024, 768), flipColors = False, fps = 30):
		"""
		Writes out a video using the input frames and given eyetracking
		@param frames: 			assumed to be same fps as the eyetracking
		@param folder:			name of folder into which to write frames
		@param dataStart:		timestamp of the start of the data frames, has precedence over firstFrame
		@param firstFrame: 		the frame number in the eyetracking that the first frame of the video corresponds to
		@param outResolution: 	desired output resolution
		@param flipColors:		flip frame color channels order?
		@param fps:				fps of the frames		
		@type frames: 			[t x h x w x 3] numpy.ndarray
		@type folder:			str
		@type dataStart:		tuple<int, int, int, int>
		@type firstFrame: 		int?
		@type outResolution: 	tuple<int, int>
		@type flipColors:		bool
		@type fps:				float
		"""
		if (dataStart is None) and (firstFrame is None):
			dataStart = self.calibrator.calibrationBeginTime
		if (dataStart is not None) and (firstFrame is not None):
			print('Both dataStart and firstFrame are provided, using dataStart')

		nFrames = frames.shape[0]
		gazeLocation = self.calibrator.TransformToScreenCoordinates(trace = self.dataPupilFinder.GetTraces(fps = fps))

		if ((frames.shape[1] != outResolution[1]) or (frames.shape[2] != outResolution[0])):
			hScale = frames.shape[2] * 1.0 / outResolution[0] * 1.0
			vScale = frames.shape[1] * 1.0 / outResolution[1] * 1.0
			transformation = AffineTransform(scale = [hScale, vScale], )
		else: transformation = None

		if dataStart is not None:
			firstFrame = self.dataPupilFinder.FindOnsetFrame(dataStart[0], dataStart[1], dataStart[2], dataStart[3])

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


	def RecenterFramesToVideo(self, frames, fileName, scale = 0.125, dataStart = None, firstFrame = None,
							  flipColors = True, padValue = 0.1, gazeSize = (1024, 768), fps = 30):
		"""
		Writes out a video in which the frames are moved such that the gaze position is always
		the center
		@param frames: 		video frames
		@param fileName: 	name of output file
		@param scale: 		scale of these frames relative to the eyetracking scale (typically 1024 x 768)
		@param dataStart:	timestamp of the start of the data frames, has precedence over firstFrame
		@param firstFrame: 	the frame number in the eyetracking that the first frame of the video corresponds to
		@param flipColors:	flip frame color channel order?
		@param padValue:	range [0, 1] value to use for padding in the recentered frames
		@param gazeSize:	stimuli resolution on to which eyetracking is mapepd
		@param fps:			fps of the frames
		@type frames: 		[t x w x h x 3] array
		@type fileName: 	str
		@type scale: 		float
		@type dataStart:	tuple<int, int, int, int>
		@type firstFrame: 	int?
		@type flipColors:	bool
		@type padValue:		float
		@type gazeSize:		tuple<int, int>
		@type fps:			float
		"""
		if (dataStart is None) and (firstFrame is None):
			dataStart = self.calibrator.calibrationBeginTime
		if (dataStart is not None) and (firstFrame is not None):
			print('Both dataStart and firstFrame are provided, using dataStart')

		nFrames = frames.shape[0]
		width = frames.shape[2]
		height = frames.shape[1]
		video = cv2.VideoWriter(fileName, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (width * 2, height * 2))
		gazeLocation = numpy.nan_to_num(self.calibrator.TransformToScreenCoordinates(trace = self.dataPupilFinder.GetTraces()))

		if dataStart is not None:
			firstFrame = self.dataPupilFinder.FindOnsetFrame(dataStart[0], dataStart[1], dataStart[2], dataStart[3])

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


	def RecenterFrames(self, frames, folder, scale = 0.125, dataStart = None, firstFrame = None,
					   flipColors = False, padValue = 0.1, gazeSize = (1024, 768), fps = 30):
		"""
		Writes recentered frames to a folder
		@param frames: 		video frames
		@param folder: 		name of folder into which to write frames
		@param scale: 		scale of these frames relative to the eyetracking scale (typically 1024 x 768)
		@param dataStart:	timestamp of the start of the data frames, has precedence over firstFrame
		@param firstFrame: 	the frame number in the eyetracking that the first frame of the video corresponds to
		@param flipColors:	flip frame color channel order?
		@param padValue:	range [0, 1] value to use for padding
		@param gazeSize:	stimuli resolution on to which eyetracking is mapped
		@param fps:			fps of the frames
		@type frames: 		[t x w x h x 3] array
		@type folder: 		str
		@type scale: 		float
		@type dataStart:	tuple<int, int, int, int>
		@type firstFrame: 	int?
		@type flipColors:	bool
		@type padValue:		float
		@type gazeSize:		tuple<int, int>
		@type fps:			float
		"""
		if (dataStart is None) and (firstFrame is None):
			dataStart = self.calibrator.calibrationBeginTime
		if (dataStart is not None) and (firstFrame is not None):
			print('Both dataStart and firstFrame are provided, using dataStart')

		nFrames = frames.shape[0]
		width = frames.shape[2]
		height = frames.shape[1]
		gazeLocation = numpy.nan_to_num(self.calibrator.TransformToScreenCoordinates(trace = self.dataPupilFinder.GetTraces()))

		if dataStart is not None:
			firstFrame = self.dataPupilFinder.FindOnsetFrame(dataStart[0], dataStart[1], dataStart[2], dataStart[3])

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


	def WriteSideBySideVideo(self, frames, fileName, dataStart = None, firstDataFrame = None,
							 stimuliResolution = (1024, 768), flipColors = True, stimulusFPS = 30):
		"""
		Writes a video out with the eyetracking video and stimulus with gaze position side by side
		@param frames:					stimulus frame array
		@param fileName:				name for video file to write
		@param dataStart:				timestamp of the start of the data frames, has precedence over firstDataFrame
		@param firstDataFrame: 			the frame number in the eyetracking that the first frame of the video corresponds to
		@param stimuliResolution:		size of the stimuli
		@param flipColors:				reverse the colors dimension of the frames when writing the video?
		@param stimulusFPS:				fps of the frames
		@type frames:					[time x h x w x 3] numpy.ndarray
		@type fileName:					str
		@type dataStart:				tuple<int, int, int, int>
		@type firstDataFrame: 			int?
		@type stimuliResolution:		tuple<int, int>
		@type flipColors:				bool
		@type stimulusFPS:				int
		"""
		if (dataStart is None) and (firstDataFrame is None):
			dataStart = self.calibrator.calibrationBeginTime
		if (dataStart is not None) and (firstDataFrame is not None):
			print('Both dataStart and firstFrame are provided, using dataStart')

		nFrames = frames.shape[0]
		gazeLocation = self.calibrator.TransformToScreenCoordinates(trace = self.dataPupilFinder.filteredPupilLocations)

		if ((frames.shape[1] != stimuliResolution[1]) or (frames.shape[2] != stimuliResolution[0])):
			hScale = frames.shape[2] * 1.0 / stimuliResolution[0] * 1.0
			vScale = frames.shape[1] * 1.0 / stimuliResolution[1] * 1.0
			transformation = AffineTransform(scale = [hScale, vScale], )
		else: transformation = None

		if dataStart is not None:
			firstDataFrame = self.dataPupilFinder.FindOnsetFrame(dataStart[0], dataStart[1], dataStart[2], dataStart[3])

		eyetrackingFramesPerStimulusFrame = int(self.dataPupilFinder.fps / stimulusFPS)

		circles = self.dataPupilFinder.filteredPupilLocations.astype(numpy.int)
		video = cv2.VideoWriter(fileName, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), self.dataPupilFinder.fps,
								(stimuliResolution[0] + 320, stimuliResolution[1] if stimuliResolution[1] > 240 else 240))
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