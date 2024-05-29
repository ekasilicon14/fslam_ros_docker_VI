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

#pragma once

#include "util/NumType.h"
#include "algorithm"
#include <boost/thread.hpp>

namespace HSLAM
{

	class Frame;

	class FrameShell
	{
	public:
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
		int id;			  // INTERNAL ID, starting at zero.
		int incoming_id;  // ID passed into DSO
		double timestamp; // timestamp passed into DSO.
		size_t KfId;

		int trackingRefId;

		// Constantly adapted
		
		AffLight aff_g2l;
		bool poseValid;

		// statisitcs
		int statistics_outlierResOnThis;
		int statistics_goodResOnThis;
		int marginalizedAt;
		double movedByOpt;

		bool isKeyframe;
		bool needRefresh;

		std::shared_ptr<Frame> frame;

		inline FrameShell()
		{
			id = 0;
			KfId = 0;
			poseValid = true;
			camToWorld = SE3();
			aff_g2l = AffLight(0,0);
			worldToCamOpti = Sim3();
			worldToCamOptiInv = Sim3();
			timestamp = 0;
			marginalizedAt = -1;
			movedByOpt = 0;
			statistics_outlierResOnThis = statistics_goodResOnThis = 0;
			trackingRefId = 0;
			isKeyframe = false;
			needRefresh = false;
		}

		// indirect!: Additional functions needed for loop closure optimization
		SE3 getPose() {
            boost::lock_guard<boost::mutex> l(shellPoseMutex);
            return camToWorld;
        }

        void setPose(const SE3 &_Twc) {
			boost::lock_guard<boost::mutex> l(shellPoseMutex);
            camToWorld = _Twc;
			Tcw = camToWorld.inverse();
			Ow = -camToWorld.rotationMatrix() * Tcw.translation();
		}

		SE3 getPoseInverse()
		{
            boost::lock_guard<boost::mutex> l(shellPoseMutex);
			return Tcw;
		}

		Vec3 getCameraCenter()
		{
			boost::lock_guard<boost::mutex> l(shellPoseMutex);
			return Ow;
		}

		// get and write the optimized pose by loop closing
        Sim3 getPoseOpti() {
            boost::lock_guard<boost::mutex> l(shellPoseMutex);
            return worldToCamOpti;
        }

		Sim3 getPoseOptiInv() {
            boost::lock_guard<boost::mutex> l(shellPoseMutex);
            return worldToCamOptiInv;
        }

		void setPoseOpti(const Sim3 &Scw)
		{
			boost::lock_guard<boost::mutex> l(shellPoseMutex);
			worldToCamOpti = Scw;
			worldToCamOptiInv = Scw.inverse();

			camToWorld = SE3(worldToCamOptiInv.rotationMatrix(), worldToCamOptiInv.translation());
			Tcw = camToWorld.inverse();
			Ow = -camToWorld.rotationMatrix() * Tcw.translation();
			needRefresh = true;
		}

		bool doesNeedRefresh()
		{
			boost::lock_guard<boost::mutex> l(shellPoseMutex);
			return needRefresh;
		}

		void setRefresh(bool _refresh)
		{
			boost::lock_guard<boost::mutex> l(shellPoseMutex);
			needRefresh = _refresh;
		}

		private:
			boost::mutex shellPoseMutex;
			SE3 camToWorld; // Write: TRACKING, while frame is still fresh; MAPPING: only when locked [shellPoseMutex].
			Sim3 worldToCamOpti; //camToWorld.inverse
			Sim3 worldToCamOptiInv;

			SE3 Tcw; //pose inverse
    		Vec3 Ow; //camera center
};


}

