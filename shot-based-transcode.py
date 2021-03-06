#!/usr/bin/env python3

import os
from subprocess import run
import datetime
from argparse import ArgumentParser

import pandas as pd
import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy import stats

from scenedetect import VideoManager
from scenedetect import SceneManager
from scenedetect.detectors import ContentDetector

version = """\
shot-based-transcode.py 2020.11.25
"""

# Encoder options:
#         --x265              use x265 rather than x264

help = f"""\
Script to transcode a video by optimising each shot individually.

Usage: {os.path.basename(__file__)} FILE [OPTION...]

Creates an `mkv` file in the current directory. The input video will be 
converted to h.264 with a constant quality as measured by VMAF. The first 
audio track will be transcoded with up to 6 channels (5.1) at 640kb/s 
surround AC3. Any further audio tracks will be ignored. All subtitle 
tracks will also be ignored. The video will be processed in no other way, 
this includes no deinterlacing and no cropping.

Optimisation options:
-q,     --quality INT       sets the target quality (default: 85)
        --crf               use CRF rate control rather than QP
-dp,    --accuracy INT      number of decimal places of accuracy when
                                using --crf mode (default: 0)	
-g,     --guess INT         sets the initial guess (default: 30)	

VMAF options:
-m,     --model PATH        path to VMAF model 
                                (default: /usr/local/share/model/vmaf_v0.6.1.pkl)
-fs,    --frame-skip INT    skips N frames during processing (default: 2)

Scene Detect options:
-th,    --threshold FLOAT   sets the threshold for detecting a new scene (default: 30.0)

Output options:
        --skip-final-analysis
                            skips the final VMAF calculations after the final encode

Other options:
-h,     --help              print this message and exit
        --version           print version information and exit
    
Requires `ffmpeg` with VMAF support
"""

class Video:
	
	def __init__(self, videoFile, encoder, crf, accuracy):
		self.videoFile = videoFile
		self.outputFile = os.path.basename(videoFile)
		self.listOfShots = []
		self.iteration = 0
		self.encoder = encoder[0]
		self.minQP = encoder[1]
		self.maxQP = encoder[2]
		self.crf = crf
		self.dp = accuracy
		

	def generateShotList(self, threshold, guess):
		print("Generating list of shots with a threshold: " + str(threshold))
		shots = self.findShots(self.videoFile, threshold)
		for shot in shots:
			self.listOfShots.append(Shot(shot, guess))


	def findShots(self, video_path, threshold):
	    # Create our video & scene managers, then add the detector.
	    video_manager = VideoManager([video_path])
	    scene_manager = SceneManager()
	    scene_manager.add_detector(ContentDetector(threshold=threshold))

	    # Base timestamp at frame 0 (required to obtain the scene list).
	    base_timecode = video_manager.get_base_timecode()

	    # Improve processing speed by downscaling before processing.
	    video_manager.set_downscale_factor()

	    # Start the video manager and perform the scene detection.
	    video_manager.start()
	    scene_manager.detect_scenes(frame_source=video_manager)

	    # Each returned scene is a tuple of the (start, end) timecode.
	    return scene_manager.get_scene_list(base_timecode)


	def sortShots(self):
		self.listOfShots = sorted(self.listOfShots, key=lambda shot: shot.firstFrame)
		
		
	def transcode(self):
		print("Transcoding ...")
		if self.encoder == "x264":
			self.transcodeX264()
		elif self.encoder == "x265":
			self.transcodeX265()
		else:
			print("Unknown encoder, again!")
			
			
	def finalTranscode(self):
		print("Transcoding ...")
		if self.encoder == "x264":
			self.finalTranscodeX264()
		elif self.encoder == "x265":
			self.finalTranscodeX265()
		else:
			print("Unknown encoder, again!")
		
		
	def transcodeX264(self):
		# Generate I-Frame and Zones strings for FFmpeg
		IFramesString = "expr:"
		zonesString = "zones="
		for shot in self.listOfShots:
			if shot.isOptimised():
				rate = self.maxQP
			else:
				rate = shot.nextQP
			
			# If rate control is qp make sure next qp is an int
			rc = "crf"
			if not self.crf:
				rate = int(rate)
				rc = "q"
				
			IFramesString += "eq(n," + str(shot.firstFrame) + ")"
			zonesString += str(shot.firstFrame) + "," + str(shot.lastFrame)
			zonesString += "," + rc + "=" + str(rate)
			if shot.firstFrame != self.listOfShots[-1].firstFrame:
				IFramesString += "+"
				zonesString += "/"
		
		# Generate complete x264 parameter string for FFmpeg		
		paramsString = "scenecut=0:"
		paramsString += zonesString

		x264 = [ "ffmpeg", "-hide_banner", "-v", "error", "-stats", "-i", self.videoFile, 
				"-y", "-force_key_frames:v", IFramesString, 
				"-c:v",	"libx264", "-preset:v", "faster", "-x264-params", paramsString, 
				"-profile:v", "high", "-color_primaries:v", "bt470bg", "-color_trc:v", 
				"bt709", "-colorspace:v", "smpte170m", "-metadata:s:v", "title\=", 
				"-disposition:v", "default", "-an", self.outputFile ]
		
		# Run FFmpeg command
		a = run(x264)
		
	 
	def finalTranscodeX264(self):
		# Generate I-Frame and Zones strings for FFmpeg
		IFramesString = "expr:"
		zonesString = "zones="
		for shot in self.listOfShots:
			rate = shot.nextQP
			
			# If rate control is qp make sure next qp is an int
			rc = "crf"
			if not self.crf:
				rate = int(rate)
				rc = "q"
				
			IFramesString += "eq(n," + str(shot.firstFrame) + ")"
			zonesString += str(shot.firstFrame) + "," + str(shot.lastFrame)
			zonesString += "," + rc + "=" + str(rate)
			if shot.firstFrame != self.listOfShots[-1].firstFrame:
				IFramesString += "+"
				zonesString += "/"
		
		# Generate complete x264 parameter string for FFmpeg		
		paramsString = "scenecut=0:"
		paramsString += zonesString
		
		# Full FFmpeg command
		x264 = [ "ffmpeg", "-hide_banner", "-v", "error", "-stats", "-i", self.videoFile, 
				"-y", "-map", "0:0", "-force_key_frames:v", IFramesString, "-c:v", 
				"libx264", "-preset:v", "medium", "-x264-params", paramsString, 
				"-profile:v", "high", "-color_primaries:v", "bt470bg", "-color_trc:v", 
				"bt709", "-colorspace:v", "smpte170m", "-metadata:s:v", "title\=", 
				"-disposition:v", "default", "-map", "0:1", "-c:a:0", "ac3", "-b:a:0", 
				"640k", "-metadata:s:a:0", "title\=", "-disposition:a:0", "default", 
				"-sn", "-metadata:g", "title\=", self.outputFile]
		
		# Run FFmpeg command
		a = run(x264)
			
	
	def transcodeX265(self):
		# Generate Zones strings for FFmpeg
		zonesString = "zones="
		for shot in self.listOfShots:
			if shot.isOptimised():
				rate = self.maxQP
			else:
				rate = shot.nextQP
			
			# If rate control is qp make sure next qp is an int
			rc = "crf"
			if not self.crf:
				rate = int(rate)
				rc = "q"
				
			zonesString += str(shot.firstFrame) + "," + str(shot.lastFrame)
			zonesString += "," + rc + "=" + str(rate)
			if shot.firstFrame != self.listOfShots[-1].firstFrame:
				zonesString += "/"
		
		# Generate complete x265 parameter string for FFmpeg		
		statsFileName = "stats-" + str(self.iteration) + ".csv"
		paramsString = "scenecut=0:csv=" + statsFileName + ":csv-log-level=1:"
		if self.iteration == 0:
			paramsString += "analysis-save=analysis.dat:analysis-save-reuse-level=10:"
		else:
			paramsString += "analysis-load=analysis.dat:analysis-load-reuse-level=10:"
		paramsString += zonesString
		
		# Full FFmpeg command
		x265 = [ "ffmpeg", "-hide_banner", "-v", "error", "-stats", "-i", self.videoFile, 
				"-y", "-c:v", "libx265", "-preset:v", "medium", 
				"-x265-params", paramsString, "-color_primaries:v", 
				"bt470bg", "-color_trc:v", "bt709", "-colorspace:v", "smpte170m", 
				"-metadata:s:v", "title\=", "-disposition:v", "default", "-an", 
				"-sn", self.outputFile]
		
		# Run FFmpeg command
		a = run(x265)
		
	
	def finalTranscodeX265(self):
		# Generate Zones strings for FFmpeg
		zonesString = "zones="
		for shot in self.listOfShots:
			rate = shot.nextQP
			
			# If rate control is qp make sure next qp is an int
			rc = "crf"
			if not self.crf:
				rate = int(rate)
				rc = "q"
				
			zonesString += str(shot.firstFrame) + "," + str(shot.lastFrame)
			zonesString += "," + rc + "=" + str(rate)
			if shot.firstFrame != self.listOfShots[-1].firstFrame:
				zonesString += "/"
		
		# Generate complete x265 parameter string for FFmpeg		
		statsFileName = "stats-" + str(self.iteration) + ".csv"
		paramsString = "scenecut=0:csv=" + statsFileName + ":csv-log-level=1:"
		if self.iteration == 0:
			paramsString += "analysis-save=analysis.dat:analysis-save-reuse-level=10:"
		else:
			paramsString += "analysis-load=analysis.dat:analysis-load-reuse-level=10:"
		paramsString += zonesString
		
		# Full FFmpeg command
		x265 = [ "ffmpeg", "-hide_banner", "-v", "error", "-stats", "-i", self.videoFile, 
				"-y", "-map", "0:0", "-c:v", "libx265", "-preset:v", "medium", 
				"-x265-params", paramsString, "-color_primaries:v", 
				"bt470bg", "-color_trc:v", "bt709", "-colorspace:v", "smpte170m", 
				"-metadata:s:v", "title\=", "-disposition:v", "default", "-map", "0:1", 
				"-c:a:0", "ac3", "-b:a:0", "640k", "-metadata:s:a:0", "title\=", 
				"-disposition:a:0", "default", "-sn", "-metadata:g", "title\=", self.outputFile]
		
		# Run FFmpeg command
		a = run(x265)
	
	
	def calculateVideoVMAF(self, modelPath):
		print("Calculating quality ...")
		
		# VMAF CSV File Name
		vmafOut = os.path.splitext(self.outputFile)[0] + "-vmaf.csv"
		
		# Assemble select, scale and vmaf filter strings		
		scaleStringMain = "[0:v]scale=1920x1080:flags=bicubic,settb=AVTB,setpts=PTS-STARTPTS[main]; "
		scaleStringRef = "[1:v]scale=1920x1080:flags=bicubic,settb=AVTB,setpts=PTS-STARTPTS[ref]; "		
		vmafFilterString = "[main][ref]libvmaf=model_path=" + modelPath
		vmafFilterString += ":log_path=" + vmafOut
		vmafFilterString += ":log_fmt=csv"
		
		# Assemble final filter string
		filterString =  scaleStringMain + scaleStringRef + vmafFilterString
		
		# Assemble FFmpeg command	
		vmafCommand = [ "ffmpeg", "-hide_banner", "-v", "error", "-stats", 
						"-r", "24", "-i", self.outputFile, 
						"-r", "24", "-i", self.videoFile, 
						"-filter_complex", filterString, "-f", "null", "-"]
		
		# Run FFmpeg command
		a = run(vmafCommand)
		
		# Read in VMAF CSV file
		vmaf_df = pd.read_csv(vmafOut, usecols=['Frame', 'vmaf'])
		
		# Averages & Percentiles
		vmafMean = vmaf_df['vmaf'].mean()
		vmafHMean = stats.hmean(vmaf_df['vmaf'],axis=0)
		vmafMax = vmaf_df['vmaf'].max()
		vmafP75 = np.percentile(vmaf_df['vmaf'], q=75)
		vmafP25 = np.percentile(vmaf_df['vmaf'], q=25)
		vmafMin = vmaf_df['vmaf'].min()
		
		# Print results
		print("VMAF mean = " + str(vmafMean))
		print("VMAF harmonic mean = " + str(vmafHMean))
		print("VMAF maximum = " + str(vmafMax))
		print("VMAF 75th percentile = " + str(vmafP75))
		print("VMAF 25th percentile = " + str(vmafP25))
		print("VMAF minimum = " + str(vmafMin))
			
				
	def calculatePerShotVmaf(self, modelPath, subSample):
		print("Calculating quality ...")
		
		# which shots need analysing
		listOfUnoptimisedShots = []
		firstFrame = 0
		lastFrame = 0
		for shot in self.listOfShots:
			if not shot.isOptimised():
				shotLength = shot.lastFrame - shot.firstFrame
				lastFrame = firstFrame + shotLength
				listOfUnoptimisedShots.append((shot, firstFrame, lastFrame))
				firstFrame = lastFrame + 1
		
		# generate selection string
		lastShotFirstFrame = listOfUnoptimisedShots[-1][0].firstFrame
		selectString = "select="
		for i in listOfUnoptimisedShots:
			shot = i[0]
			selectString += "between(n\," + str(shot.firstFrame) + "\,"
			selectString += str(shot.lastFrame) + ")"
			if shot.firstFrame != lastShotFirstFrame:
				selectString += "+"
				
		# VMAF CSV File Name
		vmafOut = os.path.splitext(self.outputFile)[0] + "-vmaf.csv"
		
		# Assemble select, scale and vmaf filter strings
		selectStringMain = "[0:v]" + selectString + "[main]; "
		selectStringRef = "[1:v]" + selectString + "[ref]; "		
		scaleStringMain = "[main]scale=1920x1080:flags=bicubic,settb=AVTB,setpts=PTS-STARTPTS[main]; "
		scaleStringRef = "[ref]scale=1920x1080:flags=bicubic,settb=AVTB,setpts=PTS-STARTPTS[ref]; "		
		vmafFilterString = "[main][ref]libvmaf=model_path=" + modelPath
		vmafFilterString += ":log_path=" + vmafOut
		vmafFilterString += ":log_fmt=csv:n_subsample=" + str(subSample+1)
		
		# Assemble final filter string
		filterString = selectStringMain + scaleStringMain + selectStringRef + \
						scaleStringRef + vmafFilterString
		
		# Assemble FFmpeg command	
		vmafCommand = [ "ffmpeg", "-hide_banner", "-v", "error", "-stats", 
						"-r", "24", "-i", self.outputFile, 
						"-r", "24", "-i", self.videoFile, 
						"-filter_complex", filterString, "-f", "null", "-"]
		
		# Run FFmpeg command
		a = run(vmafCommand)
		
		# Read in VMAF CSV file
		vmaf_df = pd.read_csv(vmafOut, usecols=['Frame', 'vmaf'])
		
		# Add QP-VMAF pairs to shot data
		for shot in listOfUnoptimisedShots:
			shotVmaf = vmaf_df.loc[(vmaf_df['Frame'] >= shot[1]) & 
									(vmaf_df['Frame'] <= shot[2])]
			
			# Calculate 25th percentile VMAF
			aveVmaf = np.percentile(shotVmaf['vmaf'], q=25)
			
			# Add QP-VMAF pair to shot object
			shot[0].addQpVmaf(shot[0].nextQP, aveVmaf)
			
		os.remove(vmafOut)
	
	
	def calculateNewSettings(self, targetVMAF):
		for shot in self.listOfShots:
			shot.calculateNewQP(targetVMAF, self.minQP, self.maxQP, self.dp)
			
	
	def printSummary(self, targetVMAF):
		shotNumber = 0
		vmaf = 0.0
		totalNumFrames = 0
		for shot in self.listOfShots:
			numFrames = shot.lastFrame - shot.firstFrame
			totalNumFrames += numFrames
			bestRQ = min(shot.qpVmaf, key=lambda rq: abs(rq[1] - targetVMAF))
			vmaf += numFrames*bestRQ[1]

			print("Shot: " + str(shotNumber))
			print("Frames: " + str(shot.firstFrame) + "-" + str(shot.lastFrame))
			print("Best: " + str(bestRQ[0]) + " - " + str(bestRQ[1]))
			print("Next: " + str(shot.nextQP))
			print("History:")
			for i in shot.qpVmaf:
				print(str(i[0]) + " - " + str(i[1]))
			print()
			
			shotNumber += 1
		print("Average VMAF: " + str(vmaf/totalNumFrames))
	
	
	def printStatus(self):
		numOptimised = 0
		for shot in self.listOfShots:
			if shot.isOptimised():
				numOptimised += 1
		print(str(numOptimised) + " optimised shots out of " + str(len(self.listOfShots)))
		
	
	def isOptimised(self):
		optimised = True
		for shot in self.listOfShots:
			if not shot.isOptimised():
				optimised = False
		return optimised
	
	
	def optimise(self, targetVMAF, modelPath, subSample):
		print("Starting optimising for VMAF: " + str(targetVMAF))
		
		while not self.isOptimised():
			print("Iteration: " + str(self.iteration+1))
			self.transcode()
			
			#self.calculateVMAF(modelPath, subSample)
			self.calculatePerShotVmaf(modelPath, subSample)
			
			print("Calculating new setting")
			self.calculateNewSettings(targetVMAF)
			
			#self.printSummary(targetVMAF)
			self.printStatus()
			self.iteration += 1
					
		print("Finished")
		

class Shot:
	
	def __init__(self, scene_list, guess):
		if scene_list[0].get_frames() == 0:
			self.firstFrame = scene_list[0].get_frames()
		else:
			self.firstFrame = scene_list[0].get_frames()+1
		self.lastFrame = scene_list[1].get_frames()
		self.nextQP = guess
		self.qpVmaf = []
	
	
	def calculateNewQP(self, targetVMAF, minQP, maxQP, dp):
		# Make sure there is some data to process
		numDataPoints = len(self.qpVmaf)
		if numDataPoints == 0:
			print("Error: Should not have called calculateNewQP before first transcode")
			exit()
			
		# If the most recent VMAF is 100.0 add 1/4th of total QP range to QP
		maxQpStep = int(np.ceil((maxQP-minQP)/4.0))
		if [x[1] for x in self.qpVmaf].count(100.0) == numDataPoints:
			if self.nextQP + maxQpStep <= maxQP:
				nextQP = self.nextQP + maxQpStep
			else:
				nextQP = maxQP
			self.nextQP = nextQP
			return 1
		
		# Sort qpVmaf by QP from low to high
		qpVmafSorted = sorted(self.qpVmaf, key=lambda row: row[0])
		
		# "t" is the transition point in QP where VMAF stops equalling 100.
		# First guess is at 0
		t = 0
		
		# If there are multiple VMAF=100 entries remove all but the one with
		# the largest QP
		while [x[1] for x in qpVmafSorted].count(100.0) > 1:
			qpVmafSorted.pop(0)
			
		# If vmaf==100 is in the data, store, set "t" as that qp value then remove it
		if [x[1] for x in qpVmafSorted].count(100.0) == 1:
			t = qpVmafSorted[0][0]
			qpVmafSorted.pop(0)
		
		# "t" provides us the minimum QP we can usefully use for a new guess (anything
		# smaller would produce vmaf=100.0)
		if t > minQP:
			minQP = t
		
		# If there are other QP's with VMAF's too large they can provide a better lower
		# bound for minQP
		for rq in qpVmafSorted:
			if rq[1] > targetVMAF and rq[0] > minQP:
				minQP = rq[0]
			
		# The maximum QP can be lowered to the smallest QP in the qpVmaf array so long
		# as the VMAF is smaller than targetVMAF
		for rq in qpVmafSorted:
			if rq[1] < targetVMAF:
				maxQP = rq[0]
				break
		
		# Use minQP and maxQP to construct array of all allowable QPs
		allowableQPs = np.arange(minQP, maxQP+1, (1.0/(10**dp)))
			
		# Separate into x's and y's and recalculate number of data points
		x = [x[0] for x in qpVmafSorted]
		y = [x[1] for x in qpVmafSorted]
		numDataPoints = len(x)
		
		# If only one data point, assume curve fits to y = a(x-t)^4 + e and e=100.
		# Else use UnivariateSpline function
		if numDataPoints == 1:
			a = (y[0]-100.0)/((x[0]-t)**4)
			y2 = [(a*(x-t)**4 + 100.0) for x in allowableQPs]
		else:
			spline = UnivariateSpline(x,y, k=1)
			y2 = [spline(x) for x in allowableQPs]
			
		# Use qp-vmaf curve to predict QP value
		qpVmafPredictions = list(zip(allowableQPs, y2))
	
		# Select value closest to targetVMAF. Then limit the size of QP jump. Then round.
		nextQP = min(qpVmafPredictions, key=lambda l: abs(l[1] - targetVMAF))[0]
		nextQP = min(nextQP, self.nextQP + maxQpStep)
		nextQP = round(nextQP, dp)
		self.nextQP = nextQP
		return 1
		
		
	def addQpVmaf(self, QP, VMAF):
		# If neither QP nor VMAF exist in qpVmaf list then add them
		# If it is an existing QP but a new VMAF, update the VMAF value in the qpVmaf list
		newQP = True
		newVMAF = True
		oldEntry = ()
		for rq in self.qpVmaf:
			if rq[0] == QP:
				newQP = False
			if rq[1] == VMAF:
				newVMAF = False
				oldEntry = rq
		if newQP and newVMAF:
			self.qpVmaf.append((QP,VMAF))
		elif newVMAF:
			self.qpVmaf.remove(oldEntry)
			self.qpVmaf.append((QP,VMAF))
	
	
	def isOptimised(self):
		optimised = True
		if len(self.qpVmaf) == 0:
				optimised = False
		else:
			QPs = [rq[0] for rq in self.qpVmaf]
			if self.nextQP not in QPs:
				optimised = False
		return optimised



def main():
	# Store start time
	startTime = datetime.datetime.now()
	
	# Create arguments
	parser = ArgumentParser(add_help=False)
	parser.add_argument("file", nargs="?")
	#parser.add_argument("--x265", action="store_true", default=False)
	parser.add_argument("-q", "--quality", type=float, default=85.0)
	parser.add_argument("--crf", action="store_true", default=False)
	parser.add_argument("-dp", "--accuracy", type=int, default=0)
	parser.add_argument("-g", "--guess", type=int, default=30)
	
	parser.add_argument("-th", "--threshold", type=float, default=30.0)
	
	parser.add_argument("-m", "--model", type=str, default="/usr/local/share/model/vmaf_v0.6.1.pkl")
	parser.add_argument("-fs", "--frame-skip", type=int, default=2)
	
	parser.add_argument("--skip-final-analysis", action="store_true", default=False)
	
	parser.add_argument("-h", "--help", action="store_true", default=False)
	parser.add_argument("--version", action="store_true", default=False)
	
	args = parser.parse_args()
	
	# Print version and exit
	if args.version:
		print(version)
		exit()
	
	# Print help and exit
	if args.help:
		print(help)
		exit()
		
	# Check file name
	if os.path.exists(args.file):
		fileName = os.path.abspath(args.file)
	else:
		print("Error, file " + str(args.file) + " does not exist")
		exit()
		
	# Select correct encoder tuple (encoderName, minQP, maxQP)
	encoder = ("x264", 10, 51)
	#if args.x265:
	#	encoder = ("x265", 0, 61)
	
	# Validate complex arguments
	validateArguments(args, encoder)
		
	# Instantiate Video class
	video = Video(fileName, encoder, args.crf, args.accuracy)
	
	# Generate and order shots
	video.generateShotList(args.threshold, args.guess)
	video.sortShots()
	
	# Optimise for quality (re-transcode until suitable VMAF achieved)
	video.optimise(args.quality, args.model, args.frame_skip)
	
	# Print details
	#video.printSummary(args.quality)
	
	# Final transcode
	video.finalTranscode()
	
	# Optional final quality check
	if not args.skip_final_analysis:
		video.calculateVideoVMAF(args.model)
	
	# Store end time & print total time
	endTime = datetime.datetime.now()
	print("Encoding time: " + str(endTime-startTime))
	

def validateArguments(args, encoder):
	
	# Check for valid quality targetVMAF
	if not 0.0 < args.quality < 100.0:
		print("Invalid target quality value: " + str(args.quality))
		print("Target quality value has to be between 0.0 and 100.0")
		print("Recommended target quality values are between 70.0 and 95.0")
		exit()
	elif not 70.0 <= args.quality <= 95.0:
		print("Target quality value " + str(args.quality) + " is valid")
		print("However a target quality value are between 70.0 and 95.0 is recommended")
		
	# Check for valid accuracy
	if not args.crf:
		args.accuracy = 0
		print("Accuracy has no effect unless you use --crf")
	if args.accuracy < 0:
		print("Invalid accuracy value: " + str(args.accuracy))
		print("Accuracy has to be positive.")
		exit()
	if args.accuracy > 3:
		print("Accuracy value " + str(args.accuracy) + " is valid")
		print("However an accuracy value greater than 3 is not recommended")
	
	# Check for valid initial guess
	allowableQPs = np.arange(encoder[1], encoder[2]+1, 1)
	if args.guess not in allowableQPs:
		print("Initial guess QP value " + str(args.guess) + " is invalid")
		print("Allowable values are whole numbers between " + str(encoder[1]) + " and " + str(encoder[2]))
		exit()
		
	# Check for valid threshold
	if args.threshold > 100.0 or args.threshold < 0.0:
		print("Invalid threshold value: " + str(args.threshold))
		print("Threshold has to be between 0.0 and 100.0")
		exit()
		
	
if __name__ == "__main__":
	main()

