/**
* This file is part of FLSAM_ROS.
Based on and inspired by DSO project by Jakob Engel

*/

#include <thread>
#include <locale.h>
#include <signal.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>



#include "IOWrapper/Output3DWrapper.h"

#include <boost/thread.hpp>
#include "util/settings.h"
#include "FullSystem/FullSystem.h"
#include "util/Undistort.h"
#include "IOWrapper/Pangolin/PangolinDSOViewer.h"
#include "IOWrapper/OutputWrapper/SampleOutputWrapper.h"


#include <ros/ros.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <geometry_msgs/PoseStamped.h>
#include "cv_bridge/cv_bridge.h"

using namespace HSLAM;

std::string calib = "";
std::string vignetteFile = "";
std::string gammaFile = "";
std::string saveFile = "";
std::string vocabPath = "";
bool useSampleOutput=false;
bool useColour = false;
int mode = 1;
int preset= 1;


void parseArgument(char* arg)
{
	int option;
	char buf[1000];
	if(1==sscanf(arg,"savefile=%s",buf))
	{
		saveFile = buf;
		printf("saving to %s on finish!\n", saveFile.c_str());
		return;
	}

	if(1==sscanf(arg,"sampleoutput=%d",&option))
	{
		if(option==1)
		{
			useSampleOutput = true;
			printf("USING SAMPLE OUTPUT WRAPPER!\n");
		}
		return;
	}

	if(1==sscanf(arg,"quiet=%d",&option))
	{
		if(option==1)
		{
			setting_debugout_runquiet = true;
			printf("QUIET MODE, I'll shut up!\n");
		}
		return;
	}


	if(1==sscanf(arg,"nolog=%d",&option))
	{
		if(option==1)
		{
			setting_logStuff = false;
			printf("DISABLE LOGGING!\n");
		}
		return;
	}

	if(1==sscanf(arg,"nogui=%d",&option))
	{
		if(option==1)
		{
			disableAllDisplay = true;
			printf("NO GUI!\n");
		}
		return;
	}
	if(1==sscanf(arg,"nomt=%d",&option))
	{
		if(option==1)
		{
			multiThreading = false;
			printf("NO MultiThreading!\n");
		}
		return;
	}
	if(1==sscanf(arg,"calib=%s",buf))
	{
		calib = buf;
		printf("loading calibration from %s!\n", calib.c_str());
		return;
	}
	if(1==sscanf(arg,"vignette=%s",buf))
	{
		vignetteFile = buf;
		printf("loading vignette from %s!\n", vignetteFile.c_str());
		return;
	}

	if(1==sscanf(arg,"gamma=%s",buf))
	{
		gammaFile = buf;
		printf("loading gammaCalib from %s!\n", gammaFile.c_str());
		return;
	}

	if(1==sscanf(arg,"LoopClosure=%d",&option))
	{
		if(option==1)
		{
			LoopClosure = true;
			printf("fslam_ros :LOOP CLOSURE IS TURNED ON!\n");
		}
		return;
	}

	if(1==sscanf(arg,"vocabPath=%s",buf))
	{
		vocabPath = buf;
		printf("fslam_ros : loading Vocabulary from %s!\n", vocabPath.c_str());
		return;
	}

	if(1==sscanf(arg,"colour=%d",&option))
	{
		if(option==1)
		{
			useColour = true;
			printf("Using Colour\n");
		}
		return;
	}

	if(1==sscanf(arg,"outpc=%d",&option))
	{
		if(option==1)
		{
			outputPC = true;
			printf("Outputting Pointcloud\n");
		}
		return;
	}

	if (1==sscanf(arg,"mode=%d",&option))
	{
		if(option==1)
		{
			setting_photometricCalibration = 0;
			setting_affineOptModeA = 0; //-1: fix. >=0: optimize (with prior, if > 0).
			setting_affineOptModeB = 0; //-1: fix. >=0: optimize (with prior, if > 0).
			
		}
		if(option==2)
		{
			setting_photometricCalibration = 0;
			setting_affineOptModeA = -1; //-1: fix. >=0: optimize (with prior, if > 0).
			setting_affineOptModeB = -1; //-1: fix. >=0: optimize (with prior, if > 0).
			setting_minGradHistAdd = 3;

		}
	
	}
	if (1==sscanf(arg,"preset=%d",&option))
	{
		if(option == 0 || option == 1)
		{
			printf("DEFAULT settings:\n"
					"- %s real-time enforcing\n"
					"- 2000 active points\n"
					"- 5-7 active frames\n"
					"- 1-6 LM iteration each KF\n"
					"- original image resolution\n", preset==0 ? "no " : "1x");
		}
		else if(option == 2 || option == 3)
		{
			printf("FAST settings:\n"
					"- %s real-time enforcing\n"
					"- 800 active points\n"
					"- 4-6 active frames\n"
					"- 1-4 LM iteration each KF\n"
					"- 424 x 320 image resolution\n", preset==0 ? "no " : "5x");
			setting_desiredImmatureDensity = 600;
			setting_desiredPointDensity = 800;
			setting_minFrames = 4;
			setting_maxFrames = 6;
			setting_maxOptIterations=4;
			setting_minOptIterations=1;

			benchmarkSetting_width = 424;
			benchmarkSetting_height = 320;

			setting_logStuff = false;
		}
	}

}




FullSystem* fullSystem = 0;
Undistort* undistorter = 0;
DBoW3::Vocabulary* Vocabpnt;
int frameID = 0;

void vidCb(const sensor_msgs::ImageConstPtr img)
{
	cv_bridge::CvImagePtr cv_ptr;
	if(!useColour){
		cv_ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO8);
		assert(cv_ptr->image.type() == CV_8U);
		assert(cv_ptr->image.channels() == 1);
	} else{
		cv_ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::BGR8);
		assert(cv_ptr->image.type() == CV_8UC3);
		assert(cv_ptr->image.channels() == 3);
	}


	if(setting_fullResetRequested)
	{
		std::vector<IOWrap::Output3DWrapper*> wraps = fullSystem->outputWrapper;
		delete fullSystem;
		for(IOWrap::Output3DWrapper* ow : wraps) ow->reset();
		fullSystem = new FullSystem();
		fullSystem->linearizeOperation=false;
		fullSystem->outputWrapper = wraps;
		if(LoopClosure)
		{
			fullSystem->setVocab(Vocabpnt);
			printf("Vocabulary Set\n");
		}
	    if(undistorter->photometricUndist != 0)
	    	fullSystem->setGammaFunction(undistorter->photometricUndist->getG());
		setting_fullResetRequested=false;
	}

	ImageAndExposure* undistImg;
	if (!useColour){
		MinimalImageB minImg((int)cv_ptr->image.cols, (int)cv_ptr->image.rows,(unsigned char*)cv_ptr->image.data);
		undistImg = undistorter->undistort<unsigned char>(&minImg, 1,0, 1.0f);
	} else {
		MinimalImageB* rimg; MinimalImageB* gimg; MinimalImageB* bimg;

		cv::Mat channels[3];
    	cv::split(cv_ptr->image, channels);

		rimg = new MinimalImageB(channels[2].cols, channels[2].rows);
		gimg = new MinimalImageB(channels[1].cols, channels[1].rows);
		bimg = new MinimalImageB(channels[0].cols, channels[0].rows);
		memcpy(bimg->data, channels[2].data, channels[2].rows*channels[2].cols);
		memcpy(gimg->data, channels[1].data, channels[1].rows*channels[1].cols);
		memcpy(rimg->data, channels[0].data, channels[0].rows*channels[0].cols);

		cv::Mat grey_m;
		cv::cvtColor(cv_ptr->image, grey_m, CV_BGR2GRAY);

		MinimalImageB* img = new MinimalImageB(grey_m.cols, grey_m.rows);
		memcpy(img->data, grey_m.data, grey_m.rows*grey_m.cols);

		undistImg = undistorter->undistort<unsigned char>(img,1,0,1.0f,true);
		undistorter->undistort_colour<unsigned char>(rimg, gimg, bimg, undistImg, 1, 0);

		delete rimg; delete gimg; delete bimg; delete img;
	}

	undistImg->timestamp=img->header.stamp.toSec(); // relay the timestamp to FSLAM
	fullSystem->addActiveFrame(undistImg, frameID);
	frameID++;
	delete undistImg;

}


//NA: Adding interruption code
bool interrupted = false;
void interruptHandler(int signal)
{
	    interrupted = true;
}



//boost exit handler to exit all threads
void my_exit_handler(int s)
{
	printf("Caught signal %d\n",s);
	exit(1);
}

void exitThread()
{
	struct sigaction sigIntHandler;
	sigIntHandler.sa_handler = my_exit_handler;
	sigemptyset(&sigIntHandler.sa_mask);
	sigIntHandler.sa_flags = 0;
	sigaction(SIGINT, &sigIntHandler, NULL);
	while(true) pause();
}


int main( int argc, char** argv )
{		
	boost::thread exThread = boost::thread(exitThread); // hook crtl+C.
	ros::init(argc, argv, "fslam_live");

	for(int i=1; i<argc;i++) parseArgument(argv[i]);


	setting_desiredImmatureDensity = 1000;
	setting_desiredPointDensity = 1200;
	setting_minFrames = 5;
	setting_maxFrames = 7;
	setting_maxOptIterations=4;
	setting_minOptIterations=1;
	setting_logStuff = false;
	setting_kfGlobalWeight = 1.3;


	printf("MODE WITH CALIBRATION, but without exposure times!\n");
	setting_photometricCalibration = 2;
	setting_affineOptModeA = 0;
	setting_affineOptModeB = 0;

	if(!vocabPath.empty())
	{
		Vocabpnt = new DBoW3::Vocabulary();
		Vocabpnt->load(vocabPath.c_str());
		LoopClosure = true; 
		printf("loaded Vocabulary from %s!\n", vocabPath.c_str());
		if (Vocabpnt->empty())
		{
			printf("failed to load vocabulary! Exit\n");
			exit(1);
		}
	}else{
		LoopClosure = false; 
	}

    undistorter = Undistort::getUndistorterForFile(calib, gammaFile, vignetteFile);

    setGlobalCalib(
            (int)undistorter->getSize()[0],
            (int)undistorter->getSize()[1],
            undistorter->getK().cast<float>());


    fullSystem = new FullSystem();
    fullSystem->linearizeOperation=false;
	
	if(LoopClosure)
	{
		fullSystem->setVocab(Vocabpnt);
		printf("Vocabulary Set\n");
	}
	
	IOWrap::PangolinDSOViewer* viewer = 0;
	if(!disableAllDisplay)
    {
        viewer = new IOWrap::PangolinDSOViewer(
	    		 (int)undistorter->getSize()[0],
	    		 (int)undistorter->getSize()[1]);
        fullSystem->outputWrapper.push_back(viewer);
    }

    if(useSampleOutput)
        fullSystem->outputWrapper.push_back(new IOWrap::SampleOutputWrapper());


    if(undistorter->photometricUndist != 0)
    	fullSystem->setGammaFunction(undistorter->photometricUndist->getG());

    ros::NodeHandle nh;
	//ros::Rate loop_rate(10);
    ros::Subscriber imgSub = nh.subscribe("image", 1, &vidCb);

    //NA: replacing ros_spin with interruptable sequence
    //ros::spin();
	
    signal(SIGINT, interruptHandler);

	while (ros::ok() && !interrupted) //&& frameID <999999 NA
		{	
			//printf("ROS IS OKAY!");
			//printf("FrameID: %d ",frameID);
			ros::spinOnce();
			//loop_rate.sleep();
			if(viewer!=0 && viewer->isDead)
					break;
			
			if(fullSystem->isLost)
            {
                printf("LOST!!\n");
                break;
            }
			
			
	if(fullSystem->initFailed || setting_fullResetRequested)
            {
                printf("RESETTING!\n");
				std::vector<IOWrap::Output3DWrapper*> wraps = fullSystem->outputWrapper;
				for(IOWrap::Output3DWrapper* ow : wraps) ow->reset();
				usleep(20000); //hack - wait for display wrapper to clean up.
				if(fullSystem)
				{
					delete fullSystem;
					fullSystem = nullptr;
				}
					
				fullSystem = new FullSystem();
    			fullSystem->setGammaFunction(undistorter->photometricUndist->getG());
				fullSystem->linearizeOperation = false;

				fullSystem->outputWrapper = wraps;

				setting_fullResetRequested=false;
            }
			
			
		}
	fullSystem->blockUntilMappingIsFinished();

	printf("fslam_ros main cpp has been interuppted.\n"); //debug NA
	ros::shutdown();
	ros::waitForShutdown();
	fullSystem->BAatExit();
			
	
	fullSystem->printResult("result.txt"); 
	if (outputPC) fullSystem->printPC("PC.PCD");

	//if(viewer != 0)
	//    viewer->run();
	//Clean-up and exit
    for(IOWrap::Output3DWrapper* ow : fullSystem->outputWrapper)
    {
		//printf("DELETE VIEWER IO wrapper\n");
        ow->join();
        delete ow;
    }

	if(LoopClosure)
	{
		delete Vocabpnt;
	}

	printf("DELETE FULLSYSTEM!\n");
	delete fullSystem;
	printf("DELETE Undistorter\n");
	delete undistorter;
	printf("EXIT NOW\n");
	return 0;
}

