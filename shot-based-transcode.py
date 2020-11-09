#!/usr/bin/env python3

from os.path import abspath, basename, splitext
from subprocess import run
import datetime
from argparse import ArgumentParser

import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scenedetect import VideoManager
from scenedetect import SceneManager
from scenedetect.detectors import ContentDetector

class Video:
	
	def __init__(self, videoFile):
		self.videoFile = videoFile
		self.outputFile = basename(videoFile)
		self.listOfShots = []
		self.iteration = 0


	def generateShotList(self, threshold=30.0):
		print("Generating list of shots with a threshold: " + str(threshold))
		shots = self.findShots(self.videoFile, threshold)
		for shot in shots:
			self.listOfShots.append(Shot(shot))


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
		
			
	def transcode(self, encoder="x264"):
		statsFileName = "stats-" + str(self.iteration) + ".csv"
		x264ParamsString = "scenecut=0:"
		x265ParamsString = "scenecut=0:csv=" + statsFileName + ":csv-log-level=1:"
		if self.iteration == 0:
			x265ParamsString += "analysis-save=analysis.dat:analysis-save-reuse-level=10:"
		else:
			x265ParamsString += "analysis-load=analysis.dat:analysis-load-reuse-level=10:"
		IFramesString = ""
		zonesString = "zones="
		lastShotFirstFrame = self.listOfShots[-1].firstFrame
		
		for shot in self.listOfShots:
			qp = shot.currentQP
			IFramesString += str(shot.firstTime)
			zonesString += str(shot.firstFrame) + "," + str(shot.lastFrame) + ",q=" + str(qp)
			if shot.firstFrame != lastShotFirstFrame:
				IFramesString += ","
				zonesString += "/"
				
		x264ParamsString += zonesString
		x265ParamsString += zonesString
		
		x264 = [ "ffmpeg", "-hide_banner", "-v", "error", "-stats", "-i", self.videoFile, 
				"-y", "-map", "0:0", "-force_key_frames:v", IFramesString, "-c:v", 
				"libx264", "-preset:v", "medium", "-x264-params", x264ParamsString, 
				"-profile:v", "high", "-color_primaries:v", "bt470bg", "-color_trc:v", 
				"bt709", "-colorspace:v", "smpte170m", "-metadata:s:v", "title\=", 
				"-disposition:v", "default", "-map", "0:1", "-c:a:0", "ac3", "-b:a:0", 
				"640k", "-metadata:s:a:0", "title\=", "-disposition:a:0", "default", 
				"-sn", "-metadata:g", "title\=", self.outputFile]
				
		x265 = [ "ffmpeg", "-hide_banner", "-v", "error", "-stats", "-i", self.videoFile, 
				"-y", "-map", "0:0", "-c:v", "libx265", "-preset:v", "medium", 
				"-x265-params", x265ParamsString, "-color_primaries:v", 
				"bt470bg", "-color_trc:v", "bt709", "-colorspace:v", "smpte170m", 
				"-metadata:s:v", "title\=", "-disposition:v", "default", "-map", "0:1", 
				"-c:a:0", "ac3", "-b:a:0", "640k", "-metadata:s:a:0", "title\=", 
				"-disposition:a:0", "default", "-sn", "-metadata:g", "title\=", self.outputFile]
		
		if encoder == "x264":
			a = run(x264)
		elif encoder == "x265":
			a = run(x265)
		else:
			print("Unknown Encoder: " + str(encoder))
	
	
	def calculateVMAF(self, modelPath, subSample):
		# which shots need analysing
		listOfUnoptimisedShots = []
		firstFrame = 0
		lastFrame = 0
		for shot in self.listOfShots:
			optimised = False
			for rq in shot.qpVmaf:
				if rq[0] == shot.currentQP:
					optimised = True
			if not optimised:
				shotLength = shot.lastFrame - shot.firstFrame
				lastFrame = firstFrame + shotLength
				listOfUnoptimisedShots.append((shot, firstFrame, lastFrame))
				print(firstFrame, lastFrame)
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
		vmafOut = splitext(self.outputFile)[0] + "-vmaf.csv"
		
		# Assemble select, scale and vmaf filter strings
		selectStringMain = "[0:v]" + selectString + "[main]; "
		selectStringRef = "[1:v]" + selectString + "[ref]; "		
		scaleStringMain = "[main]scale=1920x1080:flags=bicubic,settb=AVTB,setpts=PTS-STARTPTS[main]; "
		scaleStringRef = "[ref]scale=1920x1080:flags=bicubic,settb=AVTB,setpts=PTS-STARTPTS[ref]; "		
		vmafFilterString = "[main][ref]libvmaf=model_path=" + modelPath
		vmafFilterString += ":log_path=" + vmafOut
		vmafFilterString += ":log_fmt=csv:n_subsample=" + str(subSample)
		
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
			shot[0].addQpVmaf(shot[0].currentQP, shotVmaf['vmaf'].mean())
	
	
	def calculateNewSettings(self, targetVMAF):
		for shot in self.listOfShots:
			shot.calculateNewQP(targetVMAF)
			
	
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
			print("Next: " + str(shot.currentQP))
			print("History:")
			for i in shot.qpVmaf:
				print(str(i[0]) + " - " + str(i[1]))
			print()
			
			shotNumber += 1
		print("Average VMAF: " + str(vmaf/totalNumFrames))
	
	
	def isOptimised(self):
		optimised = True
		for shot in self.listOfShots:
			if len(shot.qpVmaf) == 0:
				optimised = False
			else:
				qpExists = False
				for rq in shot.qpVmaf:
					if rq[0] == shot.currentQP:
						qpExists = True
				if not qpExists:
					optimised = False
		return optimised
	
	
	def optimise(self, targetVMAF, encoder, modelPath, subSample):
		print("Starting optimising for VMAF: " + str(targetVMAF))
		while self.isOptimised() == False:
			print("Iteration: " + str(self.iteration+1))
			print("Transcoding ...")
			self.transcode(encoder)
			
			print("Calculating quality ...")
			self.calculateVMAF(modelPath, subSample)
			
			print("Calculating new setting")
			self.calculateNewSettings(targetVMAF)
			
			self.printSummary(targetVMAF)
			self.iteration += 1
		
		print("Finished")
		

class Shot:
	
	def __init__(self, scene_list):
		if scene_list[0].get_frames() == 0:
			self.firstFrame = scene_list[0].get_frames()
			self.firstTime = scene_list[0].get_seconds()
		else:
			self.firstFrame = scene_list[0].get_frames()+1
			self.firstTime = scene_list[0].get_seconds()+(1.0/scene_list[0].get_framerate())
		self.lastFrame = scene_list[1].get_frames()
		self.lastTime = scene_list[1].get_seconds()
		self.qpVmaf = []
		self.finalQP = 0
		self.currentQP = 30
	
	
	def calculateNewQP(self, targetVMAF, minQP=8, maxQP=50):
		# Class instantiated with currentQP set and first transcode and quality
		# calculation happen before this function is called so qpVmaf should not
		# be empty.
		
		# If there is only one Rate-Quality pair, assume R-Q curve is quadratic
		if len(self.qpVmaf) == 1:
			x1 = self.qpVmaf[-1][0]
			y1 = self.qpVmaf[-1][1]
			if y1 >= 100.0:
				y1 = 99.9
			x2 = np.arange(minQP, maxQP, 1)
			#a = (y1-100.0)/(x1*x1)				# Quadratic fit
			#y2 = [(a*x*x + 100) for x in x2]
			#a = np.arccos(y1/100.0)/x1			# Cosine fit
			#y2 = [(100*np.cos(a*x)) for x in x2]
			a = (y1-100.0)/(x1*x1*x1*x1)		# Quartic (x^4) fit
			y2 = [(a*x*x*x*x + 100) for x in x2]
			tl = list(zip(x2, y2))
			self.currentQP = min(tl, key=lambda l: abs(l[1] - targetVMAF))[0]
			
		# Fit B-Spline to data and use B-Spline function to guess next QP
		else:
			qpVmafSorted = sorted(self.qpVmaf, key=lambda row: row[0])
			x = [x[0] for x in qpVmafSorted]
			y = [x[1] for x in qpVmafSorted]
			f = interp1d(x, y, kind="linear", bounds_error=False, fill_value="extrapolate")
			x2 = np.arange(minQP, maxQP, 1)
			tl = list(zip(x2, f(x2)))
			self.currentQP = min(tl, key=lambda l: abs(l[1] - targetVMAF))[0]
		
		
	def addQpVmaf(self, QP, VMAF):
		newQP = True
		for rq in self.qpVmaf:
			if rq[0] == QP:
				self.currentQP = QP
				newQP = False		
		if newQP:
			self.qpVmaf.append((QP,VMAF))
		


def main():
	# Store start time
	startTime = datetime.datetime.now()
	
	# Manage arguments
	parser = ArgumentParser()
	parser.add_argument("file", nargs="?")
	parser.add_argument("-q", "--quality", type=float, default=85.0)
	parser.add_argument("-e", "--encoder", type=str, default="x264")	
	parser.add_argument("-t", "--threshold", type=float, default=30.0)
	parser.add_argument("-m", "--model", type=str, default="/usr/local/share/model/vmaf_v0.6.1.pkl")
	parser.add_argument("-s", "--subsample", type=int, default=1)
	args = parser.parse_args()
	fileName = abspath(args.file)
	
	# Insatiate Video class
	video = Video(fileName)
	
	# Generate and order shots
	video.generateShotList(args.threshold)
	video.sortShots()
	
	# Optimise for quality (re-transcode until suitable VMAF achieved)
	video.optimise(args.quality, args.encoder, args.model, args.subsample)
	
	# Store end time & print total time
	endTime = datetime.datetime.now()
	print("Encoding time: " + str(endTime-startTime))
	
if __name__ == "__main__":
	main()

