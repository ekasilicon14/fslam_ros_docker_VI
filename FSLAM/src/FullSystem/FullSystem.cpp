/**
* This file is part of DSO, written by Jakob Engel.
* It has been modified by Georges Younes, Daniel Asmar, John Zelek, and Yan Song Hu
*
* Copyright 2024 University of Waterloo and American University of Beirut.
* Copyright 2016 Technical University of Munich and Intel.
* Developed by Jakob Engel <engelj at in dot tum dot de>,
* for more information see <http://vision.in.tum.de/dso>.
* If you use this code, please cite the respective publications as
* listed on the above website.
*
* DSO is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* DSO is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with DSO. If not, see <http://www.gnu.org/licenses/>.
*/

/*
 * KFBuffer.cpp
 *
 *  Created on: Jan 7, 2014
 *      Author: engelj
 */


#include "FullSystem/FullSystem.h"
 
#include "stdio.h"
#include "util/globalFuncs.h"
#include <Eigen/LU>
#include <algorithm>
#include "IOWrapper/ImageDisplay.h"
#include "util/globalCalib.h"
#include <Eigen/SVD>
#include <Eigen/Eigenvalues>
#include "FullSystem/PixelSelector.h"
#include "FullSystem/PixelSelector2.h"
#include "FullSystem/ResidualProjections.h"
#include "FullSystem/ImmaturePoint.h"

#include "FullSystem/CoarseTracker.h"
#include "FullSystem/CoarseInitializer.h"

#include "Indirect/Frame.h"
#include "Indirect/Detector.h"
#include "Indirect/MapPoint.h"
#include "Indirect/Map.h"
#include "Indirect/Matcher.h"
#include "Indirect/Optimizer.h"
#include "Indirect/LoopCloser.h"

#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>

#include "OptimizationBackend/EnergyFunctional.h"
#include "OptimizationBackend/EnergyFunctionalStructs.h"

#include "IOWrapper/Output3DWrapper.h"

#include "util/ImageAndExposure.h"

#include "Indirect/IndirectTracker.h"

#include <cmath>

namespace HSLAM
{
int FrameHessian::instanceCounter=0;
int PointHessian::instanceCounter=0;
unsigned long PointHessian::totalInstantCounter=0;
int CalibHessian::instanceCounter=0;


/**
 * @brief Construct a new FullSystem Object
 * 
 * @param linearizeOperationPassed 
 * @param imuCalibration 
 * @param imuSettings 
 */
FullSystem::FullSystem()
{

	int retstat =0;
	if(setting_logStuff)
	{

		retstat += system("rm -rf logs");
		retstat += system("mkdir logs");

		retstat += system("rm -rf mats");
		retstat += system("mkdir mats");

		calibLog = new std::ofstream();
		calibLog->open("logs/calibLog.txt", std::ios::trunc | std::ios::out);
		calibLog->precision(12);

		numsLog = new std::ofstream();
		numsLog->open("logs/numsLog.txt", std::ios::trunc | std::ios::out);
		numsLog->precision(10);

		coarseTrackingLog = new std::ofstream();
		coarseTrackingLog->open("logs/coarseTrackingLog.txt", std::ios::trunc | std::ios::out);
		coarseTrackingLog->precision(10);

		eigenAllLog = new std::ofstream();
		eigenAllLog->open("logs/eigenAllLog.txt", std::ios::trunc | std::ios::out);
		eigenAllLog->precision(10);

		eigenPLog = new std::ofstream();
		eigenPLog->open("logs/eigenPLog.txt", std::ios::trunc | std::ios::out);
		eigenPLog->precision(10);

		eigenALog = new std::ofstream();
		eigenALog->open("logs/eigenALog.txt", std::ios::trunc | std::ios::out);
		eigenALog->precision(10);

		DiagonalLog = new std::ofstream();
		DiagonalLog->open("logs/diagonal.txt", std::ios::trunc | std::ios::out);
		DiagonalLog->precision(10);

		variancesLog = new std::ofstream();
		variancesLog->open("logs/variancesLog.txt", std::ios::trunc | std::ios::out);
		variancesLog->precision(10);


		nullspacesLog = new std::ofstream();
		nullspacesLog->open("logs/nullspacesLog.txt", std::ios::trunc | std::ios::out);
		nullspacesLog->precision(10);
	}
	else
	{
		nullspacesLog=0;
		variancesLog=0;
		DiagonalLog=0;
		eigenALog=0;
		eigenPLog=0;
		eigenAllLog=0;
		numsLog=0;
		calibLog=0;
	}

	assert(retstat!=293847);



	selectionMap = new float[wG[0]*hG[0]];

	coarseDistanceMap = new CoarseDistanceMap(wG[0], hG[0]);
	coarseTracker = new CoarseTracker(wG[0], hG[0]);
	coarseTracker_forNewKF = new CoarseTracker(wG[0], hG[0]);
	coarseInitializer = new CoarseInitializer(wG[0], hG[0]);
	pixelSelector = new PixelSelector(wG[0], hG[0]);

	// indirect!: Additional classes for loop closure
	detector = std::make_shared<FeatureDetector>();
	globalMap = std::make_shared<Map>();
	matcher = std::make_shared<Matcher>();
	if(LoopClosure)
		{
		// std::make_shared will not use the eigen alignment macro and cause an alignment error
		loopCloser = std::shared_ptr<LoopCloser> (new LoopCloser(this));
	}

	Velocity = SE3();

	statistics_lastNumOptIts=0;
	statistics_numDroppedPoints=0;
	statistics_numActivatedPoints=0;
	statistics_numCreatedPoints=0;
	statistics_numForceDroppedResBwd = 0;
	statistics_numForceDroppedResFwd = 0;
	statistics_numMargResFwd = 0;
	statistics_numMargResBwd = 0;

	lastCoarseRMSE.setConstant(100);

	currentMinActDist=2;
	initialized=false;


	ef = new EnergyFunctional();
	ef->red = &this->treadReduce;

	isLost=false;
	initFailed=false;


	needNewKFAfter = -1;

	linearizeOperation=true;
	runMapping=true;
	// MAPPING THREAD IS STARTED HERE
	mappingThread = boost::thread(&FullSystem::mappingLoop, this);
	lastRefStopID = 0;

	minIdJetVisDebug = -1;
	maxIdJetVisDebug = -1;
	minIdJetVisTracker = -1;
	maxIdJetVisTracker = -1;
}

/**
 * @brief Destroy the FullSystem Object
 * 
 */
FullSystem::~FullSystem()
{
	blockUntilMappingIsFinished();

	if(setting_logStuff)
	{
		calibLog->close(); delete calibLog;
		numsLog->close(); delete numsLog;
		coarseTrackingLog->close(); delete coarseTrackingLog;
		//errorsLog->close(); delete errorsLog;
		eigenAllLog->close(); delete eigenAllLog;
		eigenPLog->close(); delete eigenPLog;
		eigenALog->close(); delete eigenALog;
		DiagonalLog->close(); delete DiagonalLog;
		variancesLog->close(); delete variancesLog;
		nullspacesLog->close(); delete nullspacesLog;

	}

	delete[] selectionMap;

	for(FrameShell* s : allFrameHistory)
	{
		delete s;
		s = nullptr;
	}
	for(FrameHessian* fh : unmappedTrackedFrames)
	{
		if (fh->shell->frame)
			fh->shell->frame.reset();
		if(fh)
		{
			delete fh;
			fh = nullptr;
		}
	}
	delete coarseDistanceMap;
	delete coarseTracker;
	delete coarseTracker_forNewKF;
	delete coarseInitializer;
	delete pixelSelector;
	delete ef;
	loopCloser.reset();
	matcher.reset();
	detector.reset();
	globalMap.reset();
}

void FullSystem::setOriginalCalib(const VecXf &originalCalib, int originalW, int originalH)
{

}

void FullSystem::setGammaFunction(float* BInv)
{
	if(BInv==0) return;

	// copy BInv.
	memcpy(Hcalib.Binv, BInv, sizeof(float)*256);


	// invert.
	for(int i=1;i<255;i++)
	{
		// find val, such that Binv[val] = i.
		// I dont care about speed for this, so do it the stupid way.

		for(int s=1;s<255;s++)
		{
			if(BInv[s] <= i && BInv[s+1] >= i)
			{
				Hcalib.B[i] = s+(i - BInv[s]) / (BInv[s+1]-BInv[s]);
				break;
			}
		}
	}
	Hcalib.B[0] = 0;
	Hcalib.B[255] = 255;
}



void FullSystem::printResult(std::string file, bool printSim)
{
	int removed = 0;
	int marginalized = 0;
	int active = 0;
	int immature = 0;

	// indirect!: Count indirect points for printing
	for (auto it : allKeyFramesHistory)
	{
		if(!it->frame)
			continue;
		auto kfPts = it->frame->getMapPointsV();
		for (int i = 0, iend = kfPts.size(); i < iend; ++i)
			if (kfPts[i])
			{
				auto status = kfPts[i]->getDirStatus(); //active, marginalized, removed
				if (status == MapPoint::active)
					active += 1;
				else if (status == MapPoint::marginalized)
					marginalized += 1;
				else if (status == MapPoint::removed)
					removed += 1;
			}
		}


	boost::unique_lock<boost::mutex> lock(trackMutex);

	std::ofstream myfile;
	myfile.open (file.c_str());
	myfile << std::setprecision(15);

	for(FrameShell* s : allFrameHistory)
	{
		if(!s->poseValid) 
			continue;

		if(setting_onlyLogKFPoses && s->marginalizedAt == s->id) 
			continue;
		SE3 Twc = s->getPose();
		// Twc = s->getPose();

		myfile << s->timestamp <<
			" " << Twc.translation().transpose()<<
			" " << Twc.so3().unit_quaternion().x()<<
			" " << Twc.so3().unit_quaternion().y()<<
			" " << Twc.so3().unit_quaternion().z()<<
			" " << Twc.so3().unit_quaternion().w() << "\n";
	}
	myfile.close();
}

void FullSystem::printPC(std::string file)
{
	boost::unique_lock<boost::mutex> lock(trackMutex);
	boost::unique_lock<boost::mutex> crlock(shellPoseMutex);

	std::ofstream myfile;
	myfile.open (file.c_str());
	myfile << std::setprecision(9);

	unsigned long totalpcs = allMargPointsHistory.size();
	//unsigned long totalpcs = allMargPointsHistory.size()+allFrameHistory.size();
	
	myfile << std::string("# .PCD v.6 - Point Cloud Data file format\n");
	myfile << std::string("FIELDS x y z rgb\n");
	myfile << std::string("SIZE 4 4 4 4\n");
	myfile << std::string("TYPE F F F F\n");
	myfile << std::string("COUNT 1 1 1 1\n");
	myfile << std::string("WIDTH ") << totalpcs << std::string("\n");
	myfile << std::string("HEIGHT 1\n");
	myfile << std::string("#VIEWPOINT 0 0 0 1 0 0 0\n");
	myfile << std::string("POINTS ") << totalpcs << std::string("\n");
	myfile << std::string("DATA ascii\n");
	
	std::unordered_map<unsigned long, PC_output>::iterator itr; 
	for (itr = allMargPointsHistory.begin(); itr != allMargPointsHistory.end(); itr++)  
	{
		PC_output tmp_PC = itr->second;
		float rgb;
		unsigned char b[] = {tmp_PC.r, tmp_PC.g, tmp_PC.b, 0};
		memcpy(&rgb, &b, sizeof(rgb));

		myfile << tmp_PC.x << " " << tmp_PC.y << " " << tmp_PC.z << " " << rgb << "\n";
	} 

	// Show trajectory in point cloud
	for (FrameShell* s : allFrameHistory)  
	{
		Sophus::SE3d camToWorld = s->getPose();
		Sophus::SE3d camToFirst = camToWorld;
		float rgb;
		unsigned char b[] = {0, 255, 0, 0};
		memcpy(&rgb, &b, sizeof(rgb));
		myfile << camToFirst.translation().x() <<
            " " << camToFirst.translation().y() <<
            " " << camToFirst.translation().z() << " " << rgb << "\n";
	} 

	myfile.close();
}

/**
 * @brief Determines best track by tracking possible poses using the coarseTracker class
 * Has three steps
 * 1. Creates a list of likely poses to test
 * 2. Passes possible poses to be tracked by the coarseTracker class till a good track is returned
 * 3. Update variables
 * 
 * If a good track cannot be found, either use the estimated pose from the IMU or assume constant motion
 * If the tracking residual grows too large, this function will kill the program
 * 
 * @param fh 						Current frame
 * @param referenceToFrameHint 		Pose derived from IMU measurements
 * @return std::pair<Vec4, bool> 	Achieved residual, flow vector, and if the tracking is good
 */
Vec5 FullSystem::trackNewCoarse(FrameHessian* fh, bool writePose)
{

	assert(allFrameHistory.size() > 0);
	// set pose initialization.



	FrameHessian* lastF = coarseTracker->lastRef;

	AffLight aff_last_2_l = AffLight(0,0);

	std::vector<SE3,Eigen::aligned_allocator<SE3>> lastF_2_fh_tries;
	if(allFrameHistory.size() == 2)
		for(unsigned int i=0;i<lastF_2_fh_tries.size();i++) lastF_2_fh_tries.push_back(SE3());
	else
	{
		FrameShell* slast = allFrameHistory[allFrameHistory.size()-2];
		FrameShell* sprelast = allFrameHistory[allFrameHistory.size()-3];
		SE3 slast_2_sprelast;
		SE3 lastF_2_slast;
		{	// lock on global pose consistency!
			// boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
			slast_2_sprelast = sprelast->getPose().inverse() * slast->getPose();
			lastF_2_slast = slast->getPose().inverse() * lastF->shell->getPose();
			aff_last_2_l = slast->aff_g2l;
		}
		SE3 fh_2_slast = slast_2_sprelast;// assumed to be the same as fh_2_slast.

		if(nIndmatches > 20 && isUsable) // indirect!: if indirect tracking is good use it as prior for Direct tracking
		{
			lastF_2_fh_tries.push_back(fh->shell->getPoseInverse() * lastF->shell->getPose());
		}
		
		// get last delta-movement.
		lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast);							// assume constant motion.
		lastF_2_fh_tries.push_back(fh_2_slast.inverse() * fh_2_slast.inverse() * lastF_2_slast);	// assume double motion (frame skipped)
		lastF_2_fh_tries.push_back(SE3::exp(fh_2_slast.log()*0.5).inverse() * lastF_2_slast); // assume half motion.
		lastF_2_fh_tries.push_back(lastF_2_slast); // assume zero motion.
        lastF_2_fh_tries.push_back(SE3()); 															// assume zero motion from KF.


		// just try a TON of different initializations (all rotations). In the end,
		// if they don't work they will only be tried on the coarsest level, which is super fast anyway.
		// also, if tracking rails here we loose, so we really, really want to avoid that.
		// for(float rotDelta=0.02; rotDelta < 0.05; rotDelta++) //rotDelta+=0.01
		// {
		// 	lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,rotDelta,0,0), Vec3(0,0,0)));			// assume constant motion.
		// 	lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,0,rotDelta,0), Vec3(0,0,0)));			// assume constant motion.
		// 	lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,0,0,rotDelta), Vec3(0,0,0)));			// assume constant motion.
		// 	lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,-rotDelta,0,0), Vec3(0,0,0)));			// assume constant motion.
		// 	lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,0,-rotDelta,0), Vec3(0,0,0)));			// assume constant motion.
		// 	lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,0,0,-rotDelta), Vec3(0,0,0)));			// assume constant motion.
		// 	lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,rotDelta,rotDelta,0), Vec3(0,0,0)));	// assume constant motion.
		// 	lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,0,rotDelta,rotDelta), Vec3(0,0,0)));	// assume constant motion.
		// 	lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,rotDelta,0,rotDelta), Vec3(0,0,0)));	// assume constant motion.
		// 	lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,-rotDelta,rotDelta,0), Vec3(0,0,0)));	// assume constant motion.
		// 	lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,0,-rotDelta,rotDelta), Vec3(0,0,0)));	// assume constant motion.
		// 	lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,-rotDelta,0,rotDelta), Vec3(0,0,0)));	// assume constant motion.
		// 	lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,rotDelta,-rotDelta,0), Vec3(0,0,0)));	// assume constant motion.
		// 	lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,0,rotDelta,-rotDelta), Vec3(0,0,0)));	// assume constant motion.
		// 	lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,rotDelta,0,-rotDelta), Vec3(0,0,0)));	// assume constant motion.
		// 	lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,-rotDelta,-rotDelta,0), Vec3(0,0,0)));	// assume constant motion.
		// 	lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,0,-rotDelta,-rotDelta), Vec3(0,0,0)));	// assume constant motion.
		// 	lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,-rotDelta,0,-rotDelta), Vec3(0,0,0)));	// assume constant motion.
		// 	lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,-rotDelta,-rotDelta,-rotDelta), Vec3(0,0,0)));	// assume constant motion.
		// 	lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,-rotDelta,-rotDelta,rotDelta), Vec3(0,0,0)));	// assume constant motion.
		// 	lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,-rotDelta,rotDelta,-rotDelta), Vec3(0,0,0)));	// assume constant motion.
		// 	lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,-rotDelta,rotDelta,rotDelta), Vec3(0,0,0)));	// assume constant motion.
		// 	lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,rotDelta,-rotDelta,-rotDelta), Vec3(0,0,0)));	// assume constant motion.
		// 	lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,rotDelta,-rotDelta,rotDelta), Vec3(0,0,0)));	// assume constant motion.
		// 	lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,rotDelta,rotDelta,-rotDelta), Vec3(0,0,0)));	// assume constant motion.
		// 	lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,rotDelta,rotDelta,rotDelta), Vec3(0,0,0)));	// assume constant motion.
		// }

		if(!slast->poseValid || !sprelast->poseValid || !lastF->shell->poseValid)
		{
			lastF_2_fh_tries.clear();
			lastF_2_fh_tries.push_back(SE3());
		}
	}


	// ============== Test all of the pose guesses ===================
	Vec3 flowVecs = Vec3(100,100,100); // Flow vector of tracked motion
	SE3 lastF_2_fh = SE3();
	AffLight aff_g2l = AffLight(0,0);


	// As long as maxResForImmediateAccept is not reached, it will continue through the options.
	// Keep track of the so-far best achieved residual for each level in achievedRes.
	// If on a coarse level, tracking is WORSE than achievedRes, we will not continue to save time.


	Vec5 achievedRes = Vec5::Constant(NAN);
	bool haveOneGood = false;
	int tryIterations=0;
	for(unsigned int i=0;i<lastF_2_fh_tries.size();i++) // Try tracking for all poses in lastF_2_fh_tries
	{
		AffLight aff_g2l_this = aff_last_2_l;
		SE3 lastF_2_fh_this = lastF_2_fh_tries[i];
		// bool trackIndirect = trackNewestCoarse(fh->shell->frame, lastF->shell->frame, Test, pyrLevelsUsed - 1);
		// Tracking is done by the coarseTracker class		
		bool trackingIsGood = coarseTracker->trackNewestCoarse(
			fh, lastF_2_fh_this, aff_g2l_this,
			pyrLevelsUsed - 1,
			achievedRes);	// in each level this has to be at least as good as the last try.
		tryIterations++;

		if(i != 0)
		{
			printf("RE-TRACK ATTEMPT %d with initOption %d and start-lvl %d (ab %f %f): %f %f %f %f %f -> %f %f %f %f %f \n",
					i,
					i, pyrLevelsUsed-1,
					aff_g2l_this.a,aff_g2l_this.b,
					achievedRes[0],
					achievedRes[1],
					achievedRes[2],
					achievedRes[3],
					achievedRes[4],
					coarseTracker->lastResiduals[0],
					coarseTracker->lastResiduals[1],
					coarseTracker->lastResiduals[2],
					coarseTracker->lastResiduals[3],
					coarseTracker->lastResiduals[4]);
		}


		// ============== Update variables if there is a good track ===================
		// Track for given motion is sucessful is:
		// 1. Tracking does not return invalid values
		// 2. Residual is finite
		// 3. Residual is smaller than last achieved residual
		if(trackingIsGood && std::isfinite((float)coarseTracker->lastResiduals[0]) && !(coarseTracker->lastResiduals[0] >=  achievedRes[0]))
		{
			//printf("take over. minRes %f -> %f!\n", achievedRes[0], coarseTracker->lastResiduals[0]);
			flowVecs = coarseTracker->lastFlowIndicators;
			aff_g2l = aff_g2l_this;
			lastF_2_fh = lastF_2_fh_this;
			haveOneGood = true;
		}

		// Set achieved residuals to new values
		if(haveOneGood)
		{
			for(int j=0;j<5;j++)
			{
				if(!std::isfinite((float)achievedRes[j]) || achievedRes[j] > coarseTracker->lastResiduals[j])
					achievedRes[j] = coarseTracker->lastResiduals[j];
			}
		}


		// Stop testing other tracks if result is good enough
        if(haveOneGood &&  achievedRes[0] < lastCoarseRMSE[0]*setting_reTrackThreshold)
            break;

	}

	if(!haveOneGood)
	{
        printf("BIG ERROR! tracking failed entirely. Take predictred pose and hope we may somehow recover.\n");
		flowVecs = Vec3(0,0,0);
		aff_g2l = aff_last_2_l;
		lastF_2_fh = lastF_2_fh_tries[0];
	}

	lastCoarseRMSE = achievedRes;

	// no lock required, as fh is not used anywhere yet.

	fh->shell->trackingRefId = lastF->shell->id;
	
	fh->shell->aff_g2l = aff_g2l;

	if(writePose || tryIterations < 2)
		fh->shell->setPose(lastF->shell->getPose() * lastF_2_fh.inverse());

	if(coarseTracker->firstCoarseRMSE < 0)
		coarseTracker->firstCoarseRMSE = achievedRes[0];

    if(!setting_debugout_runquiet)
        printf("Coarse Tracker tracked ab = %f %f (exp %f). Res %f!\n", aff_g2l.a, aff_g2l.b, fh->ab_exposure, achievedRes[0]);



	if(setting_logStuff)
	{
		(*coarseTrackingLog) << std::setprecision(16)
						<< fh->shell->id << " "
						<< fh->shell->timestamp << " "
						<< fh->ab_exposure << " "
						<< fh->shell->getPose().log().transpose() << " "
						<< aff_g2l.a << " "
						<< aff_g2l.b << " "
						<< achievedRes[0] << " "
						<< tryIterations << "\n";
	}

	
	Vec5 Output;
	Output << achievedRes[0], flowVecs[0], flowVecs[1], flowVecs[2], (double)(tryIterations > 1 ? -1.0: +1.0);
	return Output;
}

/**
 * @brief Traces all the immature points
 * 
 * Because immature points are only made for new frames,
 * the number of immature points per frame will decrease over time
 * 
 * @param fh Current frame
 */
void FullSystem::traceNewCoarse(FrameHessian* fh)
{
	boost::unique_lock<boost::mutex> lock(mapMutex);

	int trace_total=0, trace_good=0, trace_oob=0, trace_out=0, trace_skip=0, trace_badcondition=0, trace_uninitialized=0;

	Mat33f K = Mat33f::Identity();
	K(0,0) = Hcalib.fxl();
	K(1,1) = Hcalib.fyl();
	K(0,2) = Hcalib.cxl();
	K(1,2) = Hcalib.cyl();

	for(FrameHessian* host : frameHessians)	// Go through all active frames
	{

		// Get SE(3) matrix from old to new position
		SE3 hostToNew = fh->PRE_worldToCam * host->PRE_camToWorld;
		// Seperate into rotation and translation parts, multiply by K at this stage for calculational efficency
		Mat33f KRKi = K * hostToNew.rotationMatrix().cast<float>() * K.inverse();
		Vec3f Kt = K * hostToNew.translation().cast<float>();

		// Get matrix that transforms points from one frame to another
		Vec2f aff = AffLight::fromToVecExposure(host->ab_exposure, fh->ab_exposure, host->aff_g2l(), fh->aff_g2l()).cast<float>();

		for(ImmaturePoint* ph : host->immaturePoints) // For all immature points in active host frame
		{
			ph->traceOn(fh, KRKi, Kt, aff, &Hcalib, false ); // trace the immature point

			if(ph->lastTraceStatus==ImmaturePointStatus::IPS_GOOD) trace_good++;
			if(ph->lastTraceStatus==ImmaturePointStatus::IPS_BADCONDITION) trace_badcondition++;
			if(ph->lastTraceStatus==ImmaturePointStatus::IPS_OOB) trace_oob++;
			if(ph->lastTraceStatus==ImmaturePointStatus::IPS_OUTLIER) trace_out++;
			if(ph->lastTraceStatus==ImmaturePointStatus::IPS_SKIPPED) trace_skip++;
			if(ph->lastTraceStatus==ImmaturePointStatus::IPS_UNINITIALIZED) trace_uninitialized++;
			trace_total++;
		}
	}
//	printf("ADD: TRACE: %'d points. %'d (%.0f%%) good. %'d (%.0f%%) skip. %'d (%.0f%%) badcond. %'d (%.0f%%) oob. %'d (%.0f%%) out. %'d (%.0f%%) uninit.\n",
//			trace_total,
//			trace_good, 100*trace_good/(float)trace_total,
//			trace_skip, 100*trace_skip/(float)trace_total,
//			trace_badcondition, 100*trace_badcondition/(float)trace_total,
//			trace_oob, 100*trace_oob/(float)trace_total,
//			trace_out, 100*trace_out/(float)trace_total,
//			trace_uninitialized, 100*trace_uninitialized/(float)trace_total);
}




/**
 * @brief Helper function for activatePointsMT
 * 
 * Does the type conversion from ImmaturePoints to PointHessians
 * 
 * @param optimized 
 * @param toOptimize 
 * @param min 
 * @param max 
 * @param stats 
 * @param tid 
 */
void FullSystem::activatePointsMT_Reductor(
		std::vector<PointHessian*>* optimized,
		std::vector<ImmaturePoint*>* toOptimize,
		int min, int max, Vec10* stats, int tid)
{
	ImmaturePointTemporaryResidual* tr = new ImmaturePointTemporaryResidual[frameHessians.size()];
	for(int k=min;k<max;k++)
	{
		(*optimized)[k] = optimizeImmaturePoint((*toOptimize)[k],1,tr);
	}
	delete[] tr;
}



/**
 * @brief Converts immature points to active points
 * 
 */
void FullSystem::activatePointsMT()
{

	// ============== Update variables for desired point density ===================
	// Change the currentMinActDist in order to achieve the desired point number
	if(ef->nPoints < setting_desiredPointDensity*0.66)
		currentMinActDist -= 0.8;
	if(ef->nPoints < setting_desiredPointDensity*0.8)
		currentMinActDist -= 0.5;
	else if(ef->nPoints < setting_desiredPointDensity*0.9)
		currentMinActDist -= 0.2;
	else if(ef->nPoints < setting_desiredPointDensity)
		currentMinActDist -= 0.1;

	if(ef->nPoints > setting_desiredPointDensity*1.5)
		currentMinActDist += 0.8;
	if(ef->nPoints > setting_desiredPointDensity*1.3)
		currentMinActDist += 0.5;
	if(ef->nPoints > setting_desiredPointDensity*1.15)
		currentMinActDist += 0.2;
	if(ef->nPoints > setting_desiredPointDensity)
		currentMinActDist += 0.1;

	// Max and min currentMinActDist values
	if(currentMinActDist < 0) currentMinActDist = 0;
	if(currentMinActDist > 4) currentMinActDist = 4;

    if(!setting_debugout_runquiet)
        printf("SPARSITY:  MinActDist %f (need %d points, have %d points)!\n",
                currentMinActDist, (int)(setting_desiredPointDensity), ef->nPoints);



	// ============== Loop through all of the immature points in active frames ===================
	FrameHessian* newestHs = frameHessians.back();

	// Make dist map.
	coarseDistanceMap->makeK(&Hcalib);
	coarseDistanceMap->makeDistanceMap(frameHessians, newestHs);

	//coarseTracker->debugPlotDistMap("distMap");

	std::vector<ImmaturePoint*> toOptimize; toOptimize.reserve(20000);


	for(FrameHessian* host : frameHessians)		// go through all active frames
	{
		if(host == newestHs) continue; // exclude newest frame

		SE3 fhToNew = newestHs->PRE_worldToCam * host->PRE_camToWorld; // Transformation matrix from host to newest frame
		// Seperate transformation matrix to rotation and translation parts
		Mat33f KRKi = (coarseDistanceMap->K[1] * fhToNew.rotationMatrix().cast<float>() * coarseDistanceMap->Ki[0]);
		Vec3f Kt = (coarseDistanceMap->K[1] * fhToNew.translation().cast<float>());


		for(unsigned int i=0;i<host->immaturePoints.size();i+=1) // for every immature point in the frame
		{
			ImmaturePoint* ph = host->immaturePoints[i];
			ph->idxInImmaturePoints = i;

			// ============== Delete invalid immature points ===================
			// Delete points that have never been traced successfully, or that are outlier on the last trace.
			if(!std::isfinite(ph->idepth_max) || ph->lastTraceStatus == IPS_OUTLIER)
			{
//				immature_invalid_deleted++;
				// remove point.
				delete ph;
				host->immaturePoints[i] = 0;
				continue;
			}

			// Activate only if this is true.
			bool canActivate = (ph->lastTraceStatus == IPS_GOOD
					|| ph->lastTraceStatus == IPS_SKIPPED
					|| ph->lastTraceStatus == IPS_BADCONDITION
					|| ph->lastTraceStatus == IPS_OOB )
							&& ph->lastTracePixelInterval < 8
							&& ph->quality > setting_minTraceQuality
							&& (ph->idepth_max+ph->idepth_min) > 0;


			// if I cannot activate the point, skip it. Maybe also delete it.
			if(!canActivate)
			{
				// if point will be out afterwards, delete it instead.
				if(ph->host->flaggedForMarginalization || ph->lastTraceStatus == IPS_OOB)
				{
//					immature_notReady_deleted++;
					delete ph;
					host->immaturePoints[i] = 0;
				}
//				immature_notReady_skipped++;
				continue;
			}


			// ============== Add immature point to activation list if it meets conditions ===================
			// Determine if the point is far away enough from other points
			// Points are only activated if it has good spacing compared to other active points
			// once projected on the newest keyframe
			Vec3f ptp = KRKi * Vec3f(ph->u, ph->v, 1) + Kt*(0.5f*(ph->idepth_max+ph->idepth_min));
			int u = ptp[0] / ptp[2] + 0.5f;
			int v = ptp[1] / ptp[2] + 0.5f;

			if((u > 0 && v > 0 && u < wG[1] && v < hG[1])) // delete point if it is out of bounds
			{

				float dist = coarseDistanceMap->fwdWarpedIDDistFinal[u+wG[1]*v] + (ptp[0]-floorf((float)(ptp[0])));

				// indirect!: Also account for indirect map points
				if(dist>=currentMinActDist* ((ph->my_type <= 4) ? ph->my_type : 1))
				{
					coarseDistanceMap->addIntoDistFinal(u,v);
					toOptimize.push_back(ph); // add point to list of points to be optimized
				}
			}
			else
			{
				delete ph;
				host->immaturePoints[i] = 0;
			}
		}
	}


//	printf("ACTIVATE: %d. (del %d, notReady %d, marg %d, good %d, marg-skip %d)\n",
//			(int)toOptimize.size(), immature_deleted, immature_notReady, immature_needMarg, immature_want, immature_margskip);

	std::vector<PointHessian*> optimized; optimized.resize(toOptimize.size());

	// ============== Activate points in activation list ===================
	// Activate points by optimizing them and converting them into PointHessians
	if(multiThreading)
		treadReduce.reduce(boost::bind(&FullSystem::activatePointsMT_Reductor, this, &optimized, &toOptimize, _1, _2, _3, _4), 0, toOptimize.size(), 50);

	else
		activatePointsMT_Reductor(&optimized, &toOptimize, 0, toOptimize.size(), 0, 0);


	// Check if all the points are valid after optimization
	for(unsigned k=0;k<toOptimize.size();k++)
	{
		PointHessian* newpoint = optimized[k];
		ImmaturePoint* ph = toOptimize[k];

		if(newpoint != 0 && newpoint != (PointHessian*)((long)(-1)))
		{
			// Add new point into the optimization and active point list
			newpoint->host->immaturePoints[ph->idxInImmaturePoints]=0;
			newpoint->host->pointHessians.push_back(newpoint);

			// indirect!: Process indirect points
			if (newpoint->my_type > 4)
			{
				// if(ph->initIdepth_min > 0)
				// 	std::cout << "idepth min " << ph->initIdepth_min << " idepth max " << ph->initIdepth_max << " idepth final " << newpoint->idepth << std::endl;
				// if(ph->priorFromInd > 0.0)
				// 	newpoint->priorFromInd = ph->priorFromInd;

				{ //strategy: block new indirect mapPoint from being created:
					auto oldMp = newpoint->host->shell->frame->getMapPoint(newpoint->my_type - 5);
					if (!oldMp)
					{
						std::shared_ptr<MapPoint> pMP = std::make_shared<MapPoint>(newpoint, globalMap);
						newpoint->host->shell->frame->addMapPoint(pMP);
						newpoint->Mp = pMP;
						pMP->AddObservation(pMP->sourceFrame, pMP->index);
						globalMap->AddMapPoint(pMP);

					}
					else
					{
						// std::shared_ptr<MapPoint> pMP = std::make_shared<MapPoint>(newpoint, globalMap);
						// // newpoint->host->shell->frame->addMapPoint(pMP);
						// newpoint->Mp = pMP;
						// pMP->AddObservation(pMP->sourceFrame, pMP->index);
						// globalMap->AddMapPoint(pMP);
						// oldMp->Replace(pMP);
						// // newpoint->priorFromInd =  oldMp->getidepthHessian(); //This point hessian was traced from an indirect map Point depth prior
					}
				}

				// { //strategy 2: replace old with new mappoint depth estimate and transfer connectivity information of old!
				// 	std::shared_ptr<MapPoint> pMP = std::make_shared<MapPoint>(newpoint, globalMap);
				// 	if (!newpoint->host->shell->frame->getMapPoint(newpoint->my_type - 5))
				// 	{
				// 		newpoint->host->shell->frame->addMapPoint(pMP);
				// 		pMP->AddObservation(pMP->sourceFrame, pMP->index);
				// 	}
				// 	else
				// 	{
				// 		newpoint->host->shell->frame->getMapPoint(newpoint->my_type - 5)->Replace(pMP);
				// 		// newpoint->hasDepthPrior = true; //This point hessian was traced from an indirect map Point depth prior
				// 	}
				// 	newpoint->Mp = pMP;
				// 	globalMap->AddMapPoint(pMP);
				// }


				//Some immature points could have been initialized from a known indirect map Point that was later removed due to KF culling, use its data:
				
			}
		

			ef->insertPoint(newpoint);
			for(PointFrameResidual* r : newpoint->residuals)
				ef->insertResidual(r);
			assert(newpoint->efPoint != 0);
			delete ph;
		}
		else if(newpoint == (PointHessian*)((long)(-1)) || ph->lastTraceStatus==IPS_OOB)
		{
			delete ph;
			if(!ph->Mp.expired())
				ph->Mp.lock()->setDirStatus(MapPoint::removed);
			ph->host->immaturePoints[ph->idxInImmaturePoints] = nullptr;
		}
		else
		{
			assert(newpoint == 0 || newpoint == (PointHessian*)((long)(-1)));
		}
	}

	// Removes immature points that have been deleted from the frames
	for (FrameHessian *host : frameHessians)
	{
		for(int i=0;i<(int)host->immaturePoints.size();i++)
		{
			if(host->immaturePoints[i]==0)
			{
				host->immaturePoints[i] = host->immaturePoints.back();
				host->immaturePoints.pop_back();
				i--;
			}
		}
	}


}


void FullSystem::activatePointsOldFirst()
{
	assert(false);
}

/**
 * @brief Flags bad points for removal
 * 
 */
void FullSystem::flagPointsForRemoval()
{
	assert(EFIndicesValid);

	std::vector<FrameHessian*> fhsToKeepPoints;
	std::vector<FrameHessian*> fhsToMargPoints;

	//if(setting_margPointVisWindow>0)
	{
		for(int i=((int)frameHessians.size())-1;i>=0 && i >= ((int)frameHessians.size());i--)
			if(!frameHessians[i]->flaggedForMarginalization) fhsToKeepPoints.push_back(frameHessians[i]);

		for(int i=0; i< (int)frameHessians.size();i++)
			if(frameHessians[i]->flaggedForMarginalization) fhsToMargPoints.push_back(frameHessians[i]);
	}



	//ef->setAdjointsF();
	//ef->setDeltaF(&Hcalib);
	int flag_oob=0, flag_in=0, flag_inin=0, flag_nores=0;

	for(FrameHessian* host : frameHessians)		// go through all active frames
	{
		for(unsigned int i=0;i<host->pointHessians.size();i++)
		{
			PointHessian* ph = host->pointHessians[i];
			if(ph==0) continue;

			// Remove points that have invalid depths
			if(ph->idepth_scaled < 0 || ph->residuals.size()==0)
			{
				host->pointHessiansOut.push_back(ph);
				ph->efPoint->stateFlag = EFPointStatus::PS_DROP;
				if(!ph->Mp.expired())
					ph->Mp.lock()->setDirStatus(MapPoint::removed);
				host->pointHessians[i] = 0;
				flag_nores++;
			}
			// Remove points that are out of bounds or are in frames that are to be marginalized
			else if(ph->isOOB(fhsToKeepPoints, fhsToMargPoints) || host->flaggedForMarginalization)
			{
				flag_oob++;
				if(ph->isInlierNew()) // Case for points with low residual values
				{
					flag_in++;
					int ngoodRes=0;
					// Linearize all
					for(PointFrameResidual* r : ph->residuals)
					{
						r->resetOOB();
						r->linearize(&Hcalib);
						r->efResidual->isLinearized = false;
						r->applyRes(true);
						if(r->efResidual->isActive())
						{
							r->efResidual->fixLinearizationF(ef);
							ngoodRes++;
						}
					}
                    if(ph->idepth_hessian > setting_minIdepthH_marg)
					{
						flag_inin++;
						ph->efPoint->stateFlag = EFPointStatus::PS_MARGINALIZE;
						host->pointHessiansMarginalized.push_back(ph);
						
						// indirect!: Remove indirect point
						if(!ph->Mp.expired())
							ph->Mp.lock()->setDirStatus(MapPoint::marginalized);
					}
					else
					{
						ph->efPoint->stateFlag = EFPointStatus::PS_DROP;
						host->pointHessiansOut.push_back(ph);
						// indirect!: Remove indirect point
						if(!ph->Mp.expired())
							ph->Mp.lock()->setDirStatus(MapPoint::removed);
					}
				}
				else
				{
					host->pointHessiansOut.push_back(ph);
					ph->efPoint->stateFlag = EFPointStatus::PS_DROP;
					// indirect!: Remove indirect point
					if(!ph->Mp.expired())
						ph->Mp.lock()->setDirStatus(MapPoint::removed);

					//printf("drop point in frame %d (%d goodRes, %d activeRes)\n", ph->host->idx, ph->numGoodResiduals, (int)ph->residuals.size());
				}

				host->pointHessians[i]=0;
			}
		}


		// Remove points that are flagged for removal from the pointHessians list
		for(int i=0;i<(int)host->pointHessians.size();i++)
		{
			if(host->pointHessians[i]==0)
			{
				host->pointHessians[i] = host->pointHessians.back();
				host->pointHessians.pop_back();
				i--;
			}
		}
	}

}

/**
 * @brief Executes odometry when new frame is added
 * 
 * This function is the starting point for most of the other functions
 * 
 * The function is passed the IMU-data from the previous frame until the current frame.
 * 
 * @param image 
 * @param id 
 * @param imuData 
 * @param gtData 
 */
void FullSystem::addActiveFrame( ImageAndExposure* image, int id )
{

    if(isLost) return;
	boost::unique_lock<boost::mutex> lock(trackMutex);

	// =========================== Add new frame into allFrameHistory =========================
	// Initialize variables
	FrameHessian* fh = new FrameHessian();
	FrameShell* shell = new FrameShell();

	// indirect!: Create indirect frame
	// Indirect point are detected when frame is made
	shell->frame = std::make_shared<Frame>(image->PhoUncalibImage, detector, &Hcalib, fh, shell, globalMap);
	
	// std::cout << shell->frame->nFeatures << std::endl;
	// cv::Mat Output;
	// shell->frame->Image.convertTo(Output, CV_8UC3);
	// cv::drawKeypoints(Output, shell->frame->mvKeys, Output, cv::Scalar(0, 255, 0));
	// cv::namedWindow("test2", cv::WINDOW_KEEPRATIO);
	// cv::imshow("test2", Output);
	// cv::waitKey(1);

	shell->marginalizedAt = shell->id = allFrameHistory.size();
    shell->timestamp = image->timestamp;
    shell->incoming_id = id;
	fh->shell = shell;
	allFrameHistory.push_back(shell);


    // =========================== Place image and image settings into frame =========================
	fh->ab_exposure = image->exposure_time;
    	fh->makeImages(image->image, &Hcalib); // Image derivative and gradient is also calculated
	if(image->useColour){
		fh->makeColourImages(image->r_image, image->g_image, image->b_image);
	}



	// =========================== Process Image =========================
	if(!initialized) // Need initialization case
	{
		// =========================== Init using image if needed =========================
		if(coarseInitializer->frameID<0)	// first frame set. fh is kept by coarseInitializer.
		{

			coarseInitializer->setFirst(&Hcalib, fh);
		}
		else if(coarseInitializer->trackFrame(fh, outputWrapper))	// if SNAPPED
		{

			initializeFromInitializer(fh);
			lock.unlock();
			deliverTrackedFrame(fh, true);
		}
		else
		{
			// if still initializing
			fh->shell->poseValid = false;
			if(fh->shell->frame)
				fh->shell->frame.reset();
			delete fh;
			fh = nullptr;
		}
		return;
	}
	else // do standard front-end operation.
	{
		
		// =========================== Swap tracking reference =========================
		if(coarseTracker_forNewKF->refFrameID > coarseTracker->refFrameID)
		{
			boost::unique_lock<boost::mutex> crlock(coarseTrackerSwapMutex);
			CoarseTracker* tmp = coarseTracker; coarseTracker=coarseTracker_forNewKF; coarseTracker_forNewKF=tmp;
		}

		// Velocity = cumulativeForm();
		
		shell->setPose(mLastFrame->fs->getPose() * Velocity.inverse()); //Velocity * LastFrameTcw
		nIndmatches = 0;
		isUsable = false;
		bool computedBoW = false;
		
		
		CheckReplacedInLastFrame();

		int nMatches;
		// indirect!: Update the loop closer and indirect points
		nMatches = matcher->SearchByProjectionFrameToFrame(shell->frame, mLastFrame, 15, true);

		if (nMatches < 20)
		{
			DBoW3::Vocabulary* weakVocanpnt = loopCloser->getVocab();
			nMatches  = matcher->SearchByBoWTracking(mpReferenceKF, shell->frame, 0.7, true, shell->frame->tMapPoints, weakVocanpnt);
			computedBoW = true;
		}

		isUsable = PoseOptimization(shell->frame, &Hcalib);

		if (!isUsable && !computedBoW)
		{
			DBoW3::Vocabulary* weakVocanpnt = loopCloser->getVocab();
			nMatches  = matcher->SearchByBoWTracking(mpReferenceKF, shell->frame, 0.7, true, shell->frame->tMapPoints, weakVocanpnt);
			isUsable = PoseOptimization(shell->frame, &Hcalib);
			computedBoW = true;
		}

		nIndmatches = updatePoseOptimizationData(shell->frame, nMatches, true);

		// int nFrametoLocalMapMatches = SearchLocalPoints(shell->frame);
		// PoseOptimization(shell->frame, &Hcalib, isUsable); //isUsable
		// nIndmatches = updatePoseOptimizationData(shell->frame, nFrametoLocalMapMatches, false);


		// =========================== Do coarse tracking =========================
		// Coarse Tracking is only done on only the new frame and reference frame
		// The reference frame should be the latest keyframe
		Vec5 tres = trackNewCoarse(fh, ! (isUsable && computedBoW) );
		

		int nFrametoLocalMapMatches = SearchLocalPoints(shell->frame);
		// checkOutliers(shell->frame, &Hcalib);
		// PoseOptimization(shell->frame, &Hcalib, isUsable); //isUsable
		nIndmatches = updatePoseOptimizationData(shell->frame, nFrametoLocalMapMatches, false);

		shell->frame->mpReferenceKF = mpReferenceKF;

		if (!std::isfinite((double)tres[0]) || !std::isfinite((double)tres[1]) || !std::isfinite((double)tres[2]) || !std::isfinite((double)tres[3]))
		{
			printf("Initial Tracking failed: LOST!\n");
			isLost = true;
			return;
		}
	
		// =========================== Make keyframe or non-keyframe =========================
		bool needToMakeKF = false;
		// Decide if keyframe or non-keyframe needs to be made
		// If setting_keyframesPerSecond is set, the keyframe is made at a specific frequency
		// Otherwise, the keyframe is made depending on specific conditions

		if(setting_keyframesPerSecond > 0) // fixed keyframe rate
		{
			needToMakeKF = allFrameHistory.size() == 1 ||
						   (fh->shell->timestamp - allKeyFramesHistory.back()->timestamp) > 0.95f / setting_keyframesPerSecond;
			needToMakeKF = needToMakeKF && (tres[4] > 0.0);

		}
		else // makes keyframe under specific conditions
		{
			Vec2 refToFh = AffLight::fromToVecExposure(coarseTracker->lastRef->ab_exposure, fh->ab_exposure,
													   coarseTracker->lastRef_aff_g2l, fh->shell->aff_g2l);

			// Make keyframe if:
			// 1. If the field of view changes too much. FOV change is measured by the mean optical flow
			// 2. If motion causes occlusions and disocclusions. This is measured by mean translation flow
			// 3. If camera exposure is changed significantly.
			// 4. Residual is too high.
			// 5. If max time between keyframes is surpassed.
			// 6. The IMU system needs a keyframe.
			needToMakeKF = allFrameHistory.size() == 1 ||
						   setting_kfGlobalWeight * setting_maxShiftWeightT * sqrtf((double)tres[1]) / (wG[0] + hG[0]) +
								   setting_kfGlobalWeight * setting_maxShiftWeightR * sqrtf((double)tres[2]) / (wG[0] + hG[0]) +
								   setting_kfGlobalWeight * setting_maxShiftWeightRT * sqrtf((double)tres[3]) / (wG[0] + hG[0]) +
								   setting_kfGlobalWeight * setting_maxAffineWeight * fabs(logf((float)refToFh[0])) >
							   1 ||
						   2 * coarseTracker->firstCoarseRMSE < tres[0];
			needToMakeKF = needToMakeKF && (tres[4] > 0.0);
		}

		//if frame succesfully tracked, update global motion model and set it to become the reference frame for the next frame

		Velocity = shell->getPoseInverse() * mLastFrame->fs->getPose(); //currentTcw * LastTwc
		// vVelocity.push(Velocity);

		mLastFrame = shell->frame;



		for (IOWrap::Output3DWrapper *ow : outputWrapper)
		{
			ow->publishCamPose(fh->shell, &Hcalib);
			ow->pushLiveFrame(fh, nIndmatches);
		}
		
		lock.unlock();
		deliverTrackedFrame(fh, needToMakeKF);
		return;
	}
}


/**
 * @brief Creates a keyframe or non-keyframe
 * 
 * Also parses outputted frame for GUI and debugging
 * Helper function for addActiveFrame
 * 
 * @param fh 
 * @param needKF 
 */
void FullSystem::deliverTrackedFrame(FrameHessian* fh, bool needKF)
{


	if(linearizeOperation)
	{
		if(goStepByStep && lastRefStopID != coarseTracker->refFrameID)
		{
			MinimalImageF3 img(wG[0], hG[0], fh->dI);
			IOWrap::displayImage("frameToTrack", &img);
			while(true)
			{
				char k=IOWrap::waitKey(0);
				if(k==' ') break;
				handleKey( k );
			}
			lastRefStopID = coarseTracker->refFrameID;
		}
		else handleKey( IOWrap::waitKey(1) );



		if(needKF) makeKeyFrame(fh);
		else makeNonKeyFrame(fh);
	}
	else
	{
		boost::unique_lock<boost::mutex> lock(trackMapSyncMutex);
		unmappedTrackedFrames.push_back(fh);
		if(needKF) needNewKFAfter=fh->shell->trackingRefId;
		trackedFrameSignal.notify_all();

		while(coarseTracker_forNewKF->refFrameID == -1 && coarseTracker->refFrameID == -1 )
		{
			mappedFrameSignal.wait(lock);
		}

		lock.unlock();
	}
}

/**
 * @brief Map using the set of active frames
 * 
 * Mapping is done when a keyframe is created
 * Runs on its own thread
 * 
 */
void FullSystem::mappingLoop()
{
	boost::unique_lock<boost::mutex> lock(trackMapSyncMutex);

	while(runMapping)
	{
		while(unmappedTrackedFrames.size()==0) // do not run if there are no frames
		{
			trackedFrameSignal.wait(lock);
			if(!runMapping) return;
		}

		FrameHessian* fh = unmappedTrackedFrames.front();
		unmappedTrackedFrames.pop_front();


        // Guaranteed to make a KF for the very first two tracked frames.
		if(allKeyFramesHistory.size() <= 2)
		{
			lock.unlock();
			makeKeyFrame(fh);
			lock.lock();
			mappedFrameSignal.notify_all();
			continue;
		}

		if(unmappedTrackedFrames.size() > 3)
			needToKetchupMapping=true;


		// if there are other frames to track, do that first.
		if(unmappedTrackedFrames.size() > 0)
		{
			lock.unlock();
			makeNonKeyFrame(fh);
			lock.lock();

			if(needToKetchupMapping && unmappedTrackedFrames.size() > 0)
			{
				FrameHessian* fh = unmappedTrackedFrames.front();
				unmappedTrackedFrames.pop_front();
				{
					// boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
					fh->setEvalPT_scaled(fh->shell->getPose().inverse(),fh->shell->aff_g2l);
				}

				if(fh->shell->frame)
					fh->shell->frame.reset();
				delete fh;
				fh = nullptr;
			}
		}
		else
		{
			if(setting_realTimeMaxKF || needNewKFAfter >= frameHessians.back()->shell->id)
			{
				lock.unlock();
				makeKeyFrame(fh);
				needToKetchupMapping=false;
				lock.lock();
			}
			else
			{
				lock.unlock();
				makeNonKeyFrame(fh);
				lock.lock();
			}
		}
		mappedFrameSignal.notify_all();
	}
	printf("MAPPING FINISHED!\n");
}

void FullSystem::blockUntilMappingIsFinished()
{
	boost::unique_lock<boost::mutex> lock(trackMapSyncMutex);
	runMapping = false;
	trackedFrameSignal.notify_all();
	lock.unlock();

	// indirect!: Handle loop closure
	if (loopCloser)
	{
		loopCloser->SetFinish(true);
		// if (globalMap->NumFrames() > 4)
		// {
		// 	globalMap->lastOptimizeAllKFs();
		// }
	}

	mappingThread.join();
	printf("Mapping Thread has been joined\n"); //debugNA

}

/**
 * @brief Creates a non-keyframe
 * 
 * @param fh 
 */
void FullSystem::makeNonKeyFrame( FrameHessian* fh)
{
	// needs to be set by mapping thread. no lock required since we are in mapping thread.
	{
		// boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
		fh->setEvalPT_scaled(fh->shell->getPose().inverse(),fh->shell->aff_g2l);
	}

	traceNewCoarse(fh);

	if(fh->shell->frame)
		fh->shell->frame.reset();
	delete fh;
	fh = nullptr;
}

/**
 * @brief Create a keyframe and optimize all of the active frames
 * 
 * @param fh 
 */
void FullSystem::makeKeyFrame( FrameHessian* fh)
{
	// needs to be set by mapping thread
	{
		// boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
		fh->setEvalPT_scaled(fh->shell->getPoseInverse(),fh->shell->aff_g2l);
		fh->shell->setPoseOpti(Sim3(fh->shell->getPoseInverse().matrix()));
	}



	traceNewCoarse(fh);

	

	boost::unique_lock<boost::mutex> lock(mapMutex);
	
	// =========================== Flag Frames to be Marginalized. =========================
	flagFramesForMarginalization(fh);


	// =========================== add New Frame to Hessian Struct. =========================
	
	fh->idx = frameHessians.size();
	frameHessians.push_back(fh);
	

	fh->shell->KfId = allKeyFramesHistory.back()->KfId + 1;
	fh->frameID = allKeyFramesHistory.size();
	allKeyFramesHistory.push_back(fh->shell);
	ef->insertFrame(fh, &Hcalib);

	IndirectMapper(fh->shell->frame);

	setPrecalcValues();



	// =========================== add new residuals for old points =========================
	int numFwdResAdde=0;
	for(FrameHessian* fh1 : frameHessians)		// go through all active frames
	{
		if(fh1 == fh) continue;
		for(PointHessian* ph : fh1->pointHessians)
		{
			PointFrameResidual* r = new PointFrameResidual(ph, fh1, fh);
			r->setState(ResState::IN);
			ph->residuals.push_back(r);
			ef->insertResidual(r);
			ph->lastResiduals[1] = ph->lastResiduals[0];
			ph->lastResiduals[0] = std::pair<PointFrameResidual*, ResState>(r, ResState::IN);
			numFwdResAdde+=1;
		}
	}



	// =========================== Activate Points (& flag for marginalization). =========================
	activatePointsMT();
	ef->makeIDX();




	// =========================== OPTIMIZE ALL =========================

	fh->frameEnergyTH = frameHessians.back()->frameEnergyTH;
	float rmse = optimize(setting_maxOptIterations);





	// =========================== Figure Out if INITIALIZATION FAILED =========================
	if(allKeyFramesHistory.size() <= 4)
	{
		if(allKeyFramesHistory.size()==2 && rmse > 20*benchmark_initializerSlackFactor)
		{
			printf("I THINK INITIALIZATINO FAILED! Resetting.\n");
			initFailed=true;
		}
		if(allKeyFramesHistory.size()==3 && rmse > 13*benchmark_initializerSlackFactor)
		{
			printf("I THINK INITIALIZATINO FAILED! Resetting.\n");
			initFailed=true;
		}
		if(allKeyFramesHistory.size()==4 && rmse > 9*benchmark_initializerSlackFactor)
		{
			printf("I THINK INITIALIZATINO FAILED! Resetting.\n");
			initFailed=true;
		}
	}



    if(isLost) return;




	// =========================== REMOVE OUTLIER =========================
	removeOutliers();




	{
		boost::unique_lock<boost::mutex> crlock(coarseTrackerSwapMutex);
		coarseTracker_forNewKF->makeK(&Hcalib);
		coarseTracker_forNewKF->setCoarseTrackingRef(frameHessians);



        coarseTracker_forNewKF->debugPlotIDepthMap(&minIdJetVisTracker, &maxIdJetVisTracker, outputWrapper);
        coarseTracker_forNewKF->debugPlotIDepthMapFloat(outputWrapper);
	}


	debugPlot("post Optimize");






	// =========================== (Activate-)Marginalize Points =========================
	flagPointsForRemoval();
	ef->dropPointsF();
	getNullspaces(
			ef->lastNullspaces_pose,
			ef->lastNullspaces_scale,
			ef->lastNullspaces_affA,
			ef->lastNullspaces_affB);
	ef->marginalizePointsF();



	// =========================== add new Immature points & new residuals =========================
	makeNewTraces(fh, 0);





    for(IOWrap::Output3DWrapper* ow : outputWrapper)
    {
        ow->publishGraph(ef->connectivityMap);
        ow->publishKeyframes(frameHessians, false, &Hcalib);
		ow->publishGlobalMap(globalMap);
	}


	// =========================== Marginalize Frames =========================

	for (unsigned int i = 0; i < frameHessians.size(); i++)
		if (frameHessians[i]->flaggedForMarginalization)
		{
			marginalizeFrame(frameHessians[i]);
			i = 0;
		}

	
	// SearchInNeighbors(fh->shell->frame);
	// KeyFrameCulling(fh->shell->frame);

	// indirect!: Update indirect structs
	updateLocalKeyframes(fh->shell->frame);
	updateLocalPoints(fh->shell->frame);

	printLogLine();
	//printEigenValLine();
}


/**
 * @brief Initializes the system if coarseInitializer finds good initial systems
 * 
 * @param newFrame 
 */
void FullSystem::initializeFromInitializer(FrameHessian* newFrame)
{
	boost::unique_lock<boost::mutex> lock(mapMutex);

	// add firstframe.
	FrameHessian* firstFrame = coarseInitializer->firstFrame;
	
	{
		firstFrame->idx = frameHessians.size();
		frameHessians.push_back(firstFrame);
	}
	
	firstFrame->frameID = allKeyFramesHistory.size();
	firstFrame->shell->KfId = 0;
	newFrame->shell->KfId = 1;
	allKeyFramesHistory.push_back(firstFrame->shell);
	ef->insertFrame(firstFrame, &Hcalib);
	setPrecalcValues();

	//int numPointsTotal = makePixelStatus(firstFrame->dI, selectionMap, wG[0], hG[0], setting_desiredDensity);
	//int numPointsTotal = pixelSelector->makeMaps(firstFrame->dIp, selectionMap,setting_desiredDensity);

	firstFrame->pointHessians.reserve(wG[0]*hG[0]*0.2f);
	firstFrame->pointHessiansMarginalized.reserve(wG[0]*hG[0]*0.2f);
	firstFrame->pointHessiansOut.reserve(wG[0]*hG[0]*0.2f);


	float sumID=1e-5, numID=1e-5;
	for(int i=0;i<coarseInitializer->numPoints[0];i++)
	{
		sumID += coarseInitializer->points[0][i].iR;
		numID++;
	}
	float rescaleFactor = 1 / (sumID / numID);

	// randomly sub-select the points I need.
	float keepPercentage = setting_desiredPointDensity / coarseInitializer->numPoints[0];

    if(!setting_debugout_runquiet)
        printf("Initialization: keep %.1f%% (need %d, have %d)!\n", 100*keepPercentage,
                (int)(setting_desiredPointDensity), coarseInitializer->numPoints[0] );


	SE3 firstToNew = coarseInitializer->thisToNext;
	firstToNew.translation() /= rescaleFactor;


	// really no lock required, as we are initializing.
	{
		// boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
		firstFrame->shell->setPose(SE3());
		firstFrame->shell->aff_g2l = AffLight(0,0);
		firstFrame->setEvalPT_scaled(firstFrame->shell->getPose().inverse(),firstFrame->shell->aff_g2l);
		firstFrame->shell->setPoseOpti(Sim3(firstFrame->shell->getPoseInverse().matrix()));

		newFrame->shell->setPose(firstToNew.inverse());
		newFrame->shell->aff_g2l = AffLight(0,0);
		newFrame->setEvalPT_scaled(newFrame->shell->getPose().inverse(),newFrame->shell->aff_g2l);
	}

	for (int i = 0; i < coarseInitializer->numPoints[0]; i++)
	{

		Pnt *point = coarseInitializer->points[0] + i;
		// indirect!: Indirect point handling
		if (point->my_type <= 4)
			if (rand() / (float)RAND_MAX > keepPercentage)
				continue;

		ImmaturePoint *pt = new ImmaturePoint(point->u + 0.5f, point->v + 0.5f, firstFrame, point->my_type, &Hcalib);

		if (!std::isfinite(pt->energyTH)) { delete pt; pt = nullptr; continue; }

		pt->idepth_max = pt->idepth_min = 1;
		PointHessian *ph = new PointHessian(pt, &Hcalib);
		delete pt;

		if (!std::isfinite(ph->energyTH)){ delete ph; continue; }

		ph->setIdepthScaled(point->iR * rescaleFactor);
		ph->setIdepthZero(ph->idepth);
		ph->hasDepthPrior = true;
		ph->setPointStatus(PointHessian::ACTIVE);

		firstFrame->pointHessians.push_back(ph);
		ef->insertPoint(ph);

		// indirect!: Indirect point handling
		if (ph->my_type > 4)
		{
			std::shared_ptr<MapPoint> pMP = std::make_shared<MapPoint>(ph, globalMap);
			ph->host->shell->frame->addMapPoint(pMP);
			ph->Mp = pMP;
			pMP->AddObservation(firstFrame->shell->frame, pMP->index);
			globalMap->AddMapPoint(pMP);
		}
	}


	// indirect!: Add indirect point to global map
	globalMap->AddKeyFrame(firstFrame->shell->frame);
	globalMap->mvpKeyFrameOrigins.push_back(firstFrame->shell->frame);
	
	
	mnLastKeyFrameId = newFrame->shell->id;
	mpLastKeyFrame = newFrame->shell->frame;

	mvpLocalKeyFrames.push_back(newFrame->shell->frame);
	mvpLocalKeyFrames.push_back(firstFrame->shell->frame);
	globalMap->GetAllMapPoints(mvpLocalMapPoints);
	
	mpReferenceKF = newFrame->shell->frame;

	newFrame->shell->frame->mpReferenceKF = newFrame->shell->frame;
	mLastFrame = newFrame->shell->frame;
	globalMap->SetReferenceMapPoints(mvpLocalMapPoints);
	
	int nmatches = SearchLocalPoints(newFrame->shell->frame, 5, 0.8);
	for (int i = 0; i < newFrame->shell->frame->nFeatures; ++i)
	{
		std::shared_ptr<MapPoint> pMP = newFrame->shell->frame->tMapPoints[i];
		if(pMP)
			pMP->increaseFound();
	}

	// indirect!: Add point to loop closer
	if (loopCloser)
		loopCloser->InsertKeyFrame(firstFrame->shell->frame, globalMap->GetMaxMPid());

	initialized = true;
	printf("INITIALIZE FROM INITIALIZER (%d pts)!\n", (int)firstFrame->pointHessians.size());
}

/**
 * @brief Create new immature points
 * 
 * Only made for the newest keyframe
 * 
 * @param newFrame 
 * @param gtDepth 
 */
void FullSystem::makeNewTraces(FrameHessian* newFrame, float* gtDepth)
{
	//int numPointsTotal = makePixelStatus(newFrame->dI, selectionMap, wG[0], hG[0], setting_desiredDensity);
	int numPointsTotal = pixelSelector->makeMaps(newFrame, selectionMap,setting_desiredImmatureDensity);

	newFrame->pointHessians.reserve(numPointsTotal*1.2f);
	//fh->pointHessiansInactive.reserve(numPointsTotal*1.2f);
	newFrame->pointHessiansMarginalized.reserve(numPointsTotal*1.2f);
	newFrame->pointHessiansOut.reserve(numPointsTotal*1.2f);

	SE3 Tcw = newFrame->shell->getPoseInverse();
	for (int y = PATTERNPADDING + 1; y < hG[0] - PATTERNPADDING - 2; y++)
		for (int x = PATTERNPADDING + 1; x < wG[0] - PATTERNPADDING - 2; x++)
		{
			int i = x + y * wG[0];
			if (selectionMap[i] == 0)
				continue;

			
			ImmaturePoint *impt = new ImmaturePoint(x, y, newFrame, selectionMap[i], &Hcalib);
			
			// indirect!: Indirect point handling
			if (selectionMap[i] > 4) //getPriors
			{
				// if(newFrame->shell->frame->getMapPoint(selectionMap[i]-5))
				// {
				// 	delete impt;
				// 	continue;
				// }
				int index = selectionMap[i] - 5;
				auto pMP = newFrame->shell->frame->getMapPoint(index);
				if (pMP)
				{
					if (!pMP->isBad())
					{
						Vec3 PointinFrame =  (Tcw * pMP->getWorldPose().cast<double>());
						float invz = (1.0 / (float)PointinFrame[2]);
						if (invz > 0)
						{
							float devi = pMP->getStdDev();
							float idepthmin = invz - 15 * devi; //15
							impt->idepth_min = idepthmin > 0 ? idepthmin : 0;
							impt->idepth_max = invz + 15 * devi; //15
							
						}
					}
				}
			}

		if (!std::isfinite(impt->energyTH))
			delete impt;
		else newFrame->immaturePoints.push_back(impt);

	}

	//printf("MADE %d IMMATURE POINTS!\n", (int)newFrame->immaturePoints.size());
}



/**
 * @brief Sets pre-calculation values for the active frames
 * 
 */
void FullSystem::setPrecalcValues()
{
	for(FrameHessian* fh : frameHessians)
	{
		fh->targetPrecalc.resize(frameHessians.size());
		for(unsigned int i=0;i<frameHessians.size();i++)
			fh->targetPrecalc[i].set(fh, frameHessians[i], &Hcalib);
	}

	ef->setDeltaF(&Hcalib);
}


/**
 * @brief For debugging
 * 
 */
void FullSystem::printLogLine()
{
	if(frameHessians.size()==0) return;

    if(!setting_debugout_runquiet)
        printf("LOG %d: %.3f fine. Res: %d A, %d L, %d M; (%'d / %'d) forceDrop. a=%f, b=%f. Window %d (%d)\n",
                allKeyFramesHistory.back()->id,
                statistics_lastFineTrackRMSE,
                ef->resInA,
                ef->resInL,
                ef->resInM,
                (int)statistics_numForceDroppedResFwd,
                (int)statistics_numForceDroppedResBwd,
                allKeyFramesHistory.back()->aff_g2l.a,
                allKeyFramesHistory.back()->aff_g2l.b,
                frameHessians.back()->shell->id - frameHessians.front()->shell->id,
                (int)frameHessians.size());


	if(!setting_logStuff) return;

	if(numsLog != 0)
	{
		(*numsLog) << allKeyFramesHistory.back()->id << " "  <<
				statistics_lastFineTrackRMSE << " "  <<
				(int)statistics_numCreatedPoints << " "  <<
				(int)statistics_numActivatedPoints << " "  <<
				(int)statistics_numDroppedPoints << " "  <<
				(int)statistics_lastNumOptIts << " "  <<
				ef->resInA << " "  <<
				ef->resInL << " "  <<
				ef->resInM << " "  <<
				statistics_numMargResFwd << " "  <<
				statistics_numMargResBwd << " "  <<
				statistics_numForceDroppedResFwd << " "  <<
				statistics_numForceDroppedResBwd << " "  <<
				frameHessians.back()->aff_g2l().a << " "  <<
				frameHessians.back()->aff_g2l().b << " "  <<
				frameHessians.back()->shell->id - frameHessians.front()->shell->id << " "  <<
				(int)frameHessians.size() << " "  << "\n";
		numsLog->flush();
	}


}



/**
 * @brief For debugging the energy function
 * 
 */
void FullSystem::printEigenValLine()
{
	if(!setting_logStuff) return;
	if(ef->lastHS.rows() < 12) return;


	MatXX Hp = ef->lastHS.bottomRightCorner(ef->lastHS.cols()-CPARS,ef->lastHS.cols()-CPARS);
	MatXX Ha = ef->lastHS.bottomRightCorner(ef->lastHS.cols()-CPARS,ef->lastHS.cols()-CPARS);
	int n = Hp.cols()/8;
	assert(Hp.cols()%8==0);

	// sub-select
	for(int i=0;i<n;i++)
	{
		MatXX tmp6 = Hp.block(i*8,0,6,n*8);
		Hp.block(i*6,0,6,n*8) = tmp6;

		MatXX tmp2 = Ha.block(i*8+6,0,2,n*8);
		Ha.block(i*2,0,2,n*8) = tmp2;
	}
	for(int i=0;i<n;i++)
	{
		MatXX tmp6 = Hp.block(0,i*8,n*8,6);
		Hp.block(0,i*6,n*8,6) = tmp6;

		MatXX tmp2 = Ha.block(0,i*8+6,n*8,2);
		Ha.block(0,i*2,n*8,2) = tmp2;
	}

	VecX eigenvaluesAll = ef->lastHS.eigenvalues().real();
	VecX eigenP = Hp.topLeftCorner(n*6,n*6).eigenvalues().real();
	VecX eigenA = Ha.topLeftCorner(n*2,n*2).eigenvalues().real();
	VecX diagonal = ef->lastHS.diagonal();

	std::sort(eigenvaluesAll.data(), eigenvaluesAll.data()+eigenvaluesAll.size());
	std::sort(eigenP.data(), eigenP.data()+eigenP.size());
	std::sort(eigenA.data(), eigenA.data()+eigenA.size());

	int nz = std::max(100,setting_maxFrames*10);

	if(eigenAllLog != 0)
	{
		VecX ea = VecX::Zero(nz); ea.head(eigenvaluesAll.size()) = eigenvaluesAll;
		(*eigenAllLog) << allKeyFramesHistory.back()->id << " " <<  ea.transpose() << "\n";
		eigenAllLog->flush();
	}
	if(eigenALog != 0)
	{
		VecX ea = VecX::Zero(nz); ea.head(eigenA.size()) = eigenA;
		(*eigenALog) << allKeyFramesHistory.back()->id << " " <<  ea.transpose() << "\n";
		eigenALog->flush();
	}
	if(eigenPLog != 0)
	{
		VecX ea = VecX::Zero(nz); ea.head(eigenP.size()) = eigenP;
		(*eigenPLog) << allKeyFramesHistory.back()->id << " " <<  ea.transpose() << "\n";
		eigenPLog->flush();
	}

	if(DiagonalLog != 0)
	{
		VecX ea = VecX::Zero(nz); ea.head(diagonal.size()) = diagonal;
		(*DiagonalLog) << allKeyFramesHistory.back()->id << " " <<  ea.transpose() << "\n";
		DiagonalLog->flush();
	}

	if(variancesLog != 0)
	{
		VecX ea = VecX::Zero(nz); ea.head(diagonal.size()) = ef->lastHS.inverse().diagonal();
		(*variancesLog) << allKeyFramesHistory.back()->id << " " <<  ea.transpose() << "\n";
		variancesLog->flush();
	}

	std::vector<VecX> &nsp = ef->lastNullspaces_forLogging;
	(*nullspacesLog) << allKeyFramesHistory.back()->id << " ";
	for(unsigned int i=0;i<nsp.size();i++)
		(*nullspacesLog) << nsp[i].dot(ef->lastHS * nsp[i]) << " " << nsp[i].dot(ef->lastbS) << " " ;
	(*nullspacesLog) << "\n";
	nullspacesLog->flush();

}

/**
 * @brief For debugging the frames
 * 
 */
void FullSystem::printFrameLifetimes()
{
	if(!setting_logStuff) return;


	boost::unique_lock<boost::mutex> lock(trackMutex);

	std::ofstream* lg = new std::ofstream();
	lg->open("logs/lifetimeLog.txt", std::ios::trunc | std::ios::out);
	lg->precision(15);

	for(FrameShell* s : allFrameHistory)
	{
		(*lg) << s->id
			<< " " << s->marginalizedAt
			<< " " << s->statistics_goodResOnThis
			<< " " << s->statistics_outlierResOnThis
			<< " " << s->movedByOpt;



		(*lg) << "\n";
	}





	lg->close();
	delete lg;

}


void FullSystem::printEvalLine()
{
	return;
}


void FullSystem::IndirectMapper(std::shared_ptr<Frame> frame)
{

	for(size_t i=0, iend = frame->tMapPoints.size(); i < iend; ++i)
    {
		std::shared_ptr<MapPoint> Mp = frame->tMapPoints[i]; 
		if (Mp)
		{	if(Mp->isBad())
				continue;
			if (!Mp->isInKeyframe(frame))
			{
				frame->addMapPointMatch(Mp, i);
				Mp->AddObservation(frame, i);
				Mp->ComputeDistinctiveDescriptors(true);
				Mp->UpdateNormalAndDepth();
			}
		}
	}

	if (frame->fs->KfId == 1) //this is the second keyframe being added to the map: need to update first kf connections.
		for (auto it: globalMap->mvpKeyFrameOrigins)
			it->UpdateConnections();

	frame->UpdateConnections();
	
	globalMap->AddKeyFrame(frame);
	mpReferenceKF = frame;

	mLastFrame->mpReferenceKF = frame;

	mnLastKeyFrameId = frame->fs->id;
    mpLastKeyFrame = frame;

	if (loopCloser)
		loopCloser->InsertKeyFrame(frame, globalMap->GetMaxMPid());

}

int FullSystem::SearchLocalPoints(std::shared_ptr<Frame> frame, int th, float nnratio)
{
	int nmatches = 0;

	for (int i = 0; i < frame->nFeatures; ++i)
	{
		std::shared_ptr<MapPoint> pMP = frame->tMapPoints[i];
		if (pMP)
		{
			if (pMP->isBad())
			{
				frame->tMapPoints[i].reset(); 
			}
			else
            {
				nmatches += 1;
                pMP->increaseVisible();
                pMP->mnLastFrameSeen = frame->fs->id;
                pMP->mbTrackInView = false;
            }
		}
	}


	int nToMatch = 0;

	boost::unique_lock<boost::mutex> lock(localMapMtx);

	for (int i = 0, iend = mvpLocalMapPoints.size(); i < iend;++i)
	{
		std::shared_ptr<MapPoint> pMP = mvpLocalMapPoints[i];

        if(pMP->mnLastFrameSeen == frame->fs->id)
            continue;
        if(pMP->isBad())
            continue;
        // Project (this fills MapPoint variables for matching)
        if(frame->isInFrustum(pMP,0.5))
        {
            pMP->increaseVisible();
            nToMatch++;
        }
	}

	if (nToMatch > 0)
	{
		// If the camera has been relocalised recently, perform a coarser search
		// if (mCurrentFrame.mnId < mnLastRelocFrameId + 2)
		// 	th = 5;
		nmatches += matcher->SearchLocalMapByProjection(frame, mvpLocalMapPoints, th, nnratio);
	}
	return nmatches;
}

bool FullSystem::TrackLocalMap(std::shared_ptr<Frame> frame)
{

	SearchLocalPoints(frame);

	// Optimize Pose
	// Optimizer::PoseOptimization(&mCurrentFrame);
	int mnMatchesInliers = 0;

	// Update MapPoints Statistics
	for (int i = 0; i < frame->nFeatures; ++i)
	{
		if (frame->tMapPoints[i])
		{
			if (!frame->mvbOutlier[i])
			{
				frame->tMapPoints[i]->increaseFound();
		
					if (frame->tMapPoints[i]->getNObservations() > 0)
						mnMatchesInliers++;
			}
		}
	}

	// Decide if the tracking was succesful
	// More restrictive if there was a relocalization recently
	// if (mCurrentFrame.mnId < mnLastRelocFrameId + mMaxFrames && mnMatchesInliers < 50)
	// 	return false;

	if (mnMatchesInliers < 30)
		return false;
	else
		return true;
}

void FullSystem::updateLocalKeyframes(std::shared_ptr<Frame> frame)
{
	mvpLocalKeyFrames.clear();

	//include currently active frames
	for (auto it : frameHessians)
	{
		assert(it->shell->frame != nullptr);

		mvpLocalKeyFrames.push_back(it->shell->frame);
		it->shell->frame->mnTrackReferenceForFrame = frame->fs->id;
		mpReferenceKF = it->shell->frame; // this will settle on the latest keyframe added in the map but might be changed later if we found one with more matches
		const std::vector<std::shared_ptr<Frame>> vNeighs = it->shell->frame->GetBestCovisibilityKeyFrames(10); //2

		for (auto it2 : vNeighs)
			if (it2->mnTrackReferenceForFrame != frame->fs->id && it->shell->frame)
			{
				mvpLocalKeyFrames.push_back(it2);
				it2->mnTrackReferenceForFrame = frame->fs->id;
			}
	}

	std::map<std::shared_ptr<Frame>, int> keyframeCounter;
	auto mapPoints = frame->getMapPointsV();

	for (auto it : mapPoints)
	{
		if (!it || it->isBad())
			continue;

		const std::map<std::shared_ptr<Frame>, size_t> observations = it->GetObservations();
		for (std::map<std::shared_ptr<Frame>, size_t>::const_iterator it = observations.begin(), itend = observations.end(); it != itend; it++)
			keyframeCounter[it->first]++;
	}

	int max = 0;
	std::shared_ptr<Frame> pKFmax = nullptr;

	for (auto it : keyframeCounter)
	{
		if (it.first->isBad() || it.second < 10)
			continue;

		if (it.second > max)
		{
			max = it.second;
			pKFmax = it.first;
		}

		if (it.first->mnTrackReferenceForFrame == frame->fs->id)
			continue;

		mvpLocalKeyFrames.push_back(it.first);
		it.first->mnTrackReferenceForFrame = frame->fs->id;
	}

	if (pKFmax)
		mpReferenceKF = pKFmax;
	frame->mpReferenceKF = mpReferenceKF;
}

void FullSystem::updateLocalPoints(std::shared_ptr<Frame> frame)
{
	// Update local MapPoints:
	boost::unique_lock<boost::mutex> lock(localMapMtx);

	mvpLocalMapPoints.clear();

	for (auto itKF : mvpLocalKeyFrames)
	{
		std::shared_ptr<Frame> pKF = itKF;
		const std::vector<std::shared_ptr<MapPoint>> vpMPs = pKF->getMapPointsV();

		for (auto pMP : vpMPs)
		{
			if (!pMP || pMP->mnTrackReferenceForFrame == frame->fs->KfId)
				continue;

			if (!pMP->isBad() ) //&& pMP->checkVar()
			{
				mvpLocalMapPoints.push_back(pMP);
				pMP->mnTrackReferenceForFrame = frame->fs->KfId;
			}
		}
	}

	globalMap->SetReferenceMapPoints(mvpLocalMapPoints);
}

void FullSystem::updateLocalKeyframesOld(std::shared_ptr<Frame> frame)
{	
	//Update Local Keyframes
	// Each map point vote for the keyframes in which it has been observed
	std::map<std::shared_ptr<Frame>, int> keyframeCounter;
	auto mapPoint = frame->getMapPointsV();

	for (int i = 0; i < frame->nFeatures; ++i)
	{
		std::shared_ptr<MapPoint> pMP = mapPoint[i]; //frame->tMapPoints[i];
		if (pMP)
		{
			if (!pMP->isBad())
			{
				const std::map<std::shared_ptr<Frame>, size_t> observations = pMP->GetObservations();
				for (std::map<std::shared_ptr<Frame>, size_t>::const_iterator it = observations.begin(), itend = observations.end(); it != itend; it++)
					keyframeCounter[it->first]++;
			}
			// else
			// {
			// 	frame->tMapPoints[i] = nullptr;
			// }
		}
	}

	if (keyframeCounter.empty())
		return;

	auto sortedKfs = sortLocalKFs(keyframeCounter, true);


	int max = 0;
	std::shared_ptr<Frame> pKFmax;

	mvpLocalKeyFrames.clear();
	mvpLocalKeyFrames.reserve(3 * sortedKfs.size());
	
	// All keyframes that observe a map point are included in the local map. Also check which keyframe shares most points
	// for (std::map<std::shared_ptr<Frame>, int>::const_iterator it = sortedKfs.begin(), itEnd = sortedKfs.end(); it != itEnd; it++)
	for (std::vector<std::pair<std::shared_ptr<Frame>, int>>::const_iterator it = sortedKfs.begin(), itEnd = sortedKfs.end(); it != itEnd; it++)
	{
		if( it - sortedKfs.begin() > 80)
			break;

		std::shared_ptr<Frame> pKF = it->first;

		if (pKF->isBad())
			continue;

		if (it->second > max)
		{
			max = it->second;
			pKFmax = pKF;
		}

		mvpLocalKeyFrames.push_back(it->first);
		pKF->mnTrackReferenceForFrame = frame->fs->id;
	}

	// Include also some not-already-included keyframes that are neighbors to already-included keyframes
	for (std::vector<std::shared_ptr<Frame>>::const_iterator itKF = mvpLocalKeyFrames.begin(), itEndKF = mvpLocalKeyFrames.end(); itKF != itEndKF; itKF++)
	{
		// Limit the number of keyframes
		if (mvpLocalKeyFrames.size() > 80)
			break;

		std::shared_ptr<Frame> pKF = *itKF;

		const std::vector<std::shared_ptr<Frame>> vNeighs = pKF->GetBestCovisibilityKeyFrames(10);

		for (std::vector<std::shared_ptr<Frame>>::const_iterator itNeighKF = vNeighs.begin(), itEndNeighKF = vNeighs.end(); itNeighKF != itEndNeighKF; itNeighKF++)
		{
			std::shared_ptr<Frame> pNeighKF = *itNeighKF;
			if (!pNeighKF->isBad())
			{
				if (pNeighKF->mnTrackReferenceForFrame != frame->fs->id)
				{
					mvpLocalKeyFrames.push_back(pNeighKF);
					pNeighKF->mnTrackReferenceForFrame = frame->fs->id;
					break;
				}
			}
		}

		const std::set<std::shared_ptr<Frame>> spChilds = pKF->GetChilds();
		for (std::set<std::shared_ptr<Frame>>::const_iterator sit = spChilds.begin(), send = spChilds.end(); sit != send; sit++)
		{
			std::shared_ptr<Frame> pChildKF = *sit;
			if (!pChildKF->isBad())
			{
				if (pChildKF->mnTrackReferenceForFrame != frame->fs->id)
				{
					mvpLocalKeyFrames.push_back(pChildKF);
					pChildKF->mnTrackReferenceForFrame = frame->fs->id;
					break;
				}
			}
		}

		std::shared_ptr<Frame> pParent = pKF->GetParent();
		if (pParent)
		{
			if (pParent->mnTrackReferenceForFrame != frame->fs->id)
			{
				mvpLocalKeyFrames.push_back(pParent);
				pParent->mnTrackReferenceForFrame = frame->fs->id;
				break;
			}
		}
	}

	if (pKFmax)
	{
		mpReferenceKF = pKFmax;
		frame->mpReferenceKF = mpReferenceKF;
	}
}


void FullSystem::updateLocalPointsOld(std::shared_ptr<Frame> frame)
{
	
	// Update local MapPoints:
	boost::unique_lock<boost::mutex> lock(localMapMtx);

	mvpLocalMapPoints.clear();

	for (std::vector<std::shared_ptr<Frame>>::const_iterator itKF = mvpLocalKeyFrames.begin(), itEndKF = mvpLocalKeyFrames.end(); itKF != itEndKF; itKF++)
	{
		std::shared_ptr<Frame> pKF = *itKF;
		const std::vector<std::shared_ptr<MapPoint>> vpMPs = pKF->getMapPointsV();

		for (std::vector<std::shared_ptr<MapPoint>>::const_iterator itMP = vpMPs.begin(), itEndMP = vpMPs.end(); itMP != itEndMP; itMP++)
		{
			std::shared_ptr<MapPoint> pMP = *itMP;
			if (!pMP)
				continue;
			if (pMP->mnTrackReferenceForFrame == frame->fs->KfId)
				continue;
			if (!pMP->isBad() && pMP->checkVar())
			{
				mvpLocalMapPoints.push_back(pMP);
				pMP->mnTrackReferenceForFrame = frame->fs->KfId;
			}
		}
	}

	for (int i = 0, iend = frameHessians.size(); i < iend; ++i)
	{
		if(!frameHessians[i]->shell->frame)
			continue;


		for (int j = 0, jend = frameHessians[i]->pointHessiansMarginalized.size(); j < jend; ++j)
		{
			std::shared_ptr<MapPoint> Mp = frameHessians[i]->pointHessiansMarginalized[j]->Mp.lock();
			if (!Mp)
				continue;
			
			if (Mp->mnTrackReferenceForFrame == frame->fs->KfId)
				continue;
			if (!Mp->isBad()&& Mp->checkVar())
			{
				mvpLocalMapPoints.push_back(Mp);
				Mp->mnTrackReferenceForFrame = frame->fs->KfId;
			}

		}

		for (int j = 0, jend = frameHessians[i]->pointHessians.size(); j < jend; ++j)
		{
			std::shared_ptr<MapPoint> Mp = frameHessians[i]->pointHessians[j]->Mp.lock();
			if (!Mp)
				continue;
			
			if (Mp->mnTrackReferenceForFrame == frame->fs->KfId)
				continue;
			if (!Mp->isBad()&& Mp->checkVar())
			{
				mvpLocalMapPoints.push_back(Mp);
				Mp->mnTrackReferenceForFrame = frame->fs->KfId;
			}

		}
	}
	globalMap->SetReferenceMapPoints(mvpLocalMapPoints);
}

void FullSystem::CheckReplacedInLastFrame()
{
	for (int i = 0; i < mLastFrame->nFeatures; ++i)
	{
		std::shared_ptr<MapPoint> pMP = mLastFrame->tMapPoints[i];

		if (pMP)
		{
			std::shared_ptr<MapPoint> pRep = pMP->GetReplaced();
			if (pRep)
			{
				mLastFrame->tMapPoints[i] = pRep;
			}
		}
	}
}

int FullSystem::updatePoseOptimizationData(std::shared_ptr<Frame> frame, int & nmatches ,bool istrackingLastFrame)
{
	int nmatchesMap = 0;
	int outliers = 0;
	for (int i = 0; i < frame->nFeatures; ++i)
	{
		if (frame->tMapPoints[i])
		{
			if (istrackingLastFrame)
			{
				if (frame->mvbOutlier[i])
				{
					std::shared_ptr<MapPoint> pMP = frame->tMapPoints[i];

					frame->tMapPoints[i].reset();
					frame->mvbOutlier[i] = false;
					pMP->mbTrackInView = false;
					pMP->mnLastFrameSeen = frame->fs->id;
					nmatches--;
					outliers++;
				}

				else if (frame->tMapPoints[i]->getNObservations() > 0)
					nmatchesMap++;
			}
			else //if tracking the localmap
			{
				if (!frame->mvbOutlier[i])
				{
					frame->tMapPoints[i]->increaseFound();

					if (frame->tMapPoints[i]->getNObservations() > 0)
						nmatchesMap++;
				}
				else //stop outliers from going to the mapping thread
				{
					frame->tMapPoints[i].reset();
					outliers++;
				}
			}
		}
	}
	// std::string out = istrackingLastFrame ? "last frame " : "local map ";
	// std::cout << "rejected outliers " + out << outliers << " total matches "<< nmatchesMap<< std::endl;
	return nmatchesMap;
}

void FullSystem::SearchInNeighbors(std::shared_ptr<Frame> currKF)
{
	// Retrieve neighbor keyframes
	int nn = 3; //5 10 20
	
	const std::vector<std::shared_ptr<Frame>> vpNeighKFs = currKF->GetBestCovisibilityKeyFrames(nn);
	std::vector<std::shared_ptr<Frame>> vpTargetKFs;
	for (std::vector<std::shared_ptr<Frame>>::const_iterator vit = vpNeighKFs.begin(), vend = vpNeighKFs.end(); vit != vend; vit++)
	{
		std::shared_ptr<Frame> pKFi = *vit;
		if (pKFi->isBad() || pKFi->mnFuseTargetForKF == currKF->fs->KfId)
			continue;
		vpTargetKFs.push_back(pKFi);
		pKFi->mnFuseTargetForKF = currKF->fs->KfId;

		// Extend to some second neighbors
		// const std::vector<std::shared_ptr<Frame>> vpSecondNeighKFs = pKFi->GetBestCovisibilityKeyFrames(2); //5
		// for (std::vector<std::shared_ptr<Frame>>::const_iterator vit2 = vpSecondNeighKFs.begin(), vend2 = vpSecondNeighKFs.end(); vit2 != vend2; vit2++)
		// {
		// 	std::shared_ptr<Frame> pKFi2 = *vit2;
		// 	if (pKFi2->isBad() || pKFi2->mnFuseTargetForKF == currKF->fs->KfId || pKFi2->fs->KfId == currKF->fs->KfId)
		// 		continue;
		// 	vpTargetKFs.push_back(pKFi2);
		// }
	}

	// Search matches by projection from current KF in target KFs
	std::vector<std::shared_ptr<MapPoint>> vpMapPointMatches = currKF->getMapPointsV();
	for (std::vector<std::shared_ptr<Frame>>::iterator vit = vpTargetKFs.begin(), vend = vpTargetKFs.end(); vit != vend; vit++)
	{
		std::shared_ptr<Frame> pKFi = *vit;
		matcher->Fuse(pKFi, vpMapPointMatches, 3.0); //th = 3.0
	}
	
	// Search matches by projection from target KFs in current KF
	std::vector<std::shared_ptr<MapPoint>> vpFuseCandidates;
	vpFuseCandidates.reserve(vpTargetKFs.size() * vpMapPointMatches.size());

	for (std::vector<std::shared_ptr<Frame>>::iterator vitKF = vpTargetKFs.begin(), vendKF = vpTargetKFs.end(); vitKF != vendKF; vitKF++)
	{
		std::shared_ptr<Frame> pKFi = *vitKF;

		std::vector<std::shared_ptr<MapPoint>> vpMapPointsKFi = pKFi->getMapPointsV();

		for (std::vector<std::shared_ptr<MapPoint>>::iterator vitMP = vpMapPointsKFi.begin(), vendMP = vpMapPointsKFi.end(); vitMP != vendMP; vitMP++)
		{
			std::shared_ptr<MapPoint> pMP = *vitMP;
			if (!pMP)
				continue;
			if (pMP->isBad() || pMP->mnFuseCandidateForKF == currKF->fs->KfId)
				continue;
			pMP->mnFuseCandidateForKF = currKF->fs->KfId;
			vpFuseCandidates.push_back(pMP);
		}
	}

	matcher->Fuse(currKF, vpFuseCandidates, 3.0); //th = 3.0

	// Update points
	vpMapPointMatches = currKF->getMapPointsV();
	for (size_t i = 0, iend = vpMapPointMatches.size(); i < iend; i++)
	{
		std::shared_ptr<MapPoint> pMP = vpMapPointMatches[i];
		if (pMP)
		{
			if (!pMP->isBad())
			{
				pMP->ComputeDistinctiveDescriptors(false);
				pMP->UpdateNormalAndDepth();
			}
		}
	}

	// Update connections in covisibility graph
	currKF->UpdateConnections();
}


void FullSystem::KeyFrameCulling(std::shared_ptr<Frame> currKF)
{
    // Check redundant keyframes (only local keyframes)
    // A keyframe is considered redundant if the 80% of the MapPoints it sees, are seen
    // in at least other 3 keyframes (in the same or finer scale)
    // We only consider close stereo points
    std::vector<std::shared_ptr<Frame>> vpLocalKeyFrames = currKF->GetVectorCovisibleKeyFrames();
	int KfsChecked = 0;
	for (std::vector<std::shared_ptr<Frame>>::iterator vit = vpLocalKeyFrames.begin(), vend = vpLocalKeyFrames.end(); vit != vend; vit++)
	{	
		std::shared_ptr<Frame> pKF = *vit;
		if(pKF->fs->KfId==0 || (pKF->getState()== Frame::kfstate::active))
            continue;

		int age = mpLastKeyFrame->fs->KfId - pKF->fs->KfId;
		if (age > 20)
			continue;

		KfsChecked += 1;
		if(KfsChecked > 30)
			return;

		const std::vector<std::shared_ptr<MapPoint>> vpMapPoints = pKF->getMapPointsV();

        int nObs = 3;
        const int thObs=nObs;
        int nRedundantObservations=0;
        int nMPs=0;
        for(size_t i=0, iend=vpMapPoints.size(); i<iend; i++)
        {
            std::shared_ptr<MapPoint> pMP = vpMapPoints[i];
            if(pMP)
            {
                if(!pMP->isBad())
                {
                    nMPs++;
                    if(pMP->getNObservations()>thObs)
                    {
                        // const int &scaleLevel = pKF->mvKeysUn[i].octave;
                        const std::map<std::shared_ptr<Frame>, size_t> observations = pMP->GetObservations();
                        int nObs=0;
                        for(std::map<std::shared_ptr<Frame>, size_t>::const_iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
                        {
                            std::shared_ptr<Frame> pKFi = mit->first;
                            if(pKFi==pKF)
                                continue;
                            // const int &scaleLeveli = pKFi->mvKeysUn[mit->second].octave;

                            // if(scaleLeveli<=scaleLevel+1)
                            // {
                                nObs++;
                                if(nObs>=thObs)
                                    break;
                            // }
                        }
                        if(nObs>=thObs)
                        {
                            nRedundantObservations++;
                        }
                    }
                }
            }
        }  

        if(nRedundantObservations>0.9*nMPs) //0.9
            {
				if(pKF->getState() != Frame::kfstate::active)
				{
					pKF->setBadFlag();		
					pKF->fs->frame.reset();
				}
			}
	}
}

void FullSystem::BAatExit()
{
	std::vector<std::shared_ptr<Frame>> allKFrames;
	std::vector<std::shared_ptr<MapPoint>> allMapPoints;

	globalMap->GetAllKeyFrames(allKFrames);
	globalMap->GetAllMapPoints(allMapPoints);


	bool stopGBA = false;

	size_t currMaxKF = allKeyFramesHistory.back()->KfId;
	size_t currMaxMp = globalMap->GetMaxMPid();

	BundleAdjustment(allKFrames, allMapPoints, 10, &stopGBA, true, true, currMaxKF, currMaxKF - 15, currMaxMp);
	for (auto it : allKeyFramesHistory)
	    it->setRefresh(true);
}

void FullSystem::setVocab(DBoW3::Vocabulary* _Vocabpnt)
{
	loopCloser->lc_setVocab(_Vocabpnt);
	globalMap->m_setVocab(_Vocabpnt);
}


// SE3 FullSystem::cumulativeForm()
// {
// 	auto v = vVelocity;
// 	if (vVelocity.size() == 4)
// 	{
// 		int u = 3;

// 		SE3 t1 = v.front(); v.pop();
// 		SE3 t2 = v.front(); v.pop();
// 		SE3 t3 = v.front(); v.pop();
// 		SE3 t4 = v.front(); v.pop();
// 		return SE3::exp(t1.log()) *
// 			   SE3::exp(((5 + 3 * u - 3 * u * u + u * u * u) / 6) * SE3(t1.inverse() * t2).log()) *
// 			   SE3::exp(((1 + 3 * u + 3 * u * u - 2 * u * u * u) / 6) * SE3(t2.inverse() * t3).log()) *
// 			   SE3::exp(((u * u * u) / 6) * SE3(t3.inverse() * t4).log());
// 	}
// 	else
// 	{
// 		if(vVelocity.size() == 0 )
// 			return SE3();

// 		return vVelocity.back();
// 	}
// }
}
