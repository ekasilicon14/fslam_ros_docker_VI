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



#include "FullSystem/HessianBlocks.h"
#include "util/FrameShell.h"
#include "FullSystem/ImmaturePoint.h"
#include "OptimizationBackend/EnergyFunctionalStructs.h"
#include "Indirect/MapPoint.h"



namespace HSLAM
{


/**
 * @brief Construct a new Point Hessian:: Point Hessian object
 * 
 * Hessian component associated with one point
 * 
 * @param rawPoint 
 * @param Hcalib 
 */
PointHessian::PointHessian(const ImmaturePoint* const rawPoint, CalibHessian* Hcalib)
{
	instanceCounter++;
	point_id = totalInstantCounter;
	totalInstantCounter++;
	host = rawPoint->host; // host frame
	hasDepthPrior=false;

	idepth_hessian=0;
	maxRelBaseline=0;
	numGoodResiduals=0;

	// Set static values & initialization.
	u = rawPoint->u;
	v = rawPoint->v;
	assert(std::isfinite(rawPoint->idepth_max));
	//idepth_init = rawPoint->idepth_GT;

	// my_type is block level where point was selected
	my_type = rawPoint->my_type;

	setIdepthScaled((rawPoint->idepth_max + rawPoint->idepth_min)*0.5);
	setPointStatus(PointHessian::INACTIVE);

	int n = PATTERNNUM;
	memcpy(color, rawPoint->color, sizeof(float)*n);
	memcpy(weights, rawPoint->weights, sizeof(float)*n);
	energyTH = rawPoint->energyTH;

	colourValid = false;

	if(rawPoint->colourValid){
		colourValid = true;
		for(int i = 0; i < PATTERNNUM; i++){
			colour3[i] = rawPoint->colour3[i];
		}
	}
	efPoint=0;
}


/**
 * @brief Release residuals
 * 
 */
void PointHessian::release()
{
	for(unsigned int i=0;i<residuals.size();i++) delete residuals[i];
	residuals.clear();
}


/**
 * @brief Set zero state
 * 
 * @param state_zero 
 */
void FrameHessian::setStateZero(const Vec10 &state_zero)
{
	assert(state_zero.head<6>().squaredNorm() < 1e-20);

	this->state_zero = state_zero;


	for(int i=0;i<6;i++)
	{
		Vec6 eps; eps.setZero(); eps[i] = 1e-3;
		SE3 EepsP = Sophus::SE3d::exp(eps);
		SE3 EepsM = Sophus::SE3d::exp(-eps);
		SE3 w2c_leftEps_P_x0 = (get_worldToCam_evalPT() * EepsP) * get_worldToCam_evalPT().inverse();
		SE3 w2c_leftEps_M_x0 = (get_worldToCam_evalPT() * EepsM) * get_worldToCam_evalPT().inverse();
		nullspaces_pose.col(i) = (w2c_leftEps_P_x0.log() - w2c_leftEps_M_x0.log())/(2e-3);
	}
	//nullspaces_pose.topRows<3>() *= SCALE_XI_TRANS_INVERSE;
	//nullspaces_pose.bottomRows<3>() *= SCALE_XI_ROT_INVERSE;

	// scale change
	SE3 w2c_leftEps_P_x0 = (get_worldToCam_evalPT());
	w2c_leftEps_P_x0.translation() *= 1.00001;
	w2c_leftEps_P_x0 = w2c_leftEps_P_x0 * get_worldToCam_evalPT().inverse();
	SE3 w2c_leftEps_M_x0 = (get_worldToCam_evalPT());
	w2c_leftEps_M_x0.translation() /= 1.00001;
	w2c_leftEps_M_x0 = w2c_leftEps_M_x0 * get_worldToCam_evalPT().inverse();
	nullspaces_scale = (w2c_leftEps_P_x0.log() - w2c_leftEps_M_x0.log())/(2e-3);


	nullspaces_affine.setZero();
	nullspaces_affine.topLeftCorner<2,1>()  = Vec2(1,0);
	assert(ab_exposure > 0);
	nullspaces_affine.topRightCorner<2,1>() = Vec2(0, expf(aff_g2l_0().a)*ab_exposure);
};



void FrameHessian::release()
{
	// DELETE POINT
	// DELETE RESIDUAL

	for (unsigned int i = 0; i < pointHessians.size(); i++)
	{
		// if(pointHessians[i] && !pointHessians[i]->Mp.expired())
		// 	pointHessians[i]->Mp.lock()->setDirStatus(MapPoint::marginalized);
		delete pointHessians[i];
	}
	
	for (unsigned int i = 0; i < pointHessiansMarginalized.size(); i++)
	{
		// if(pointHessiansMarginalized[i] && !pointHessiansMarginalized[i]->Mp.expired())
		// 	pointHessiansMarginalized[i]->Mp.lock()->setDirStatus(MapPoint::marginalized);
		delete pointHessiansMarginalized[i];
	}

	for(unsigned int i=0;i<pointHessiansOut.size();i++) 
	{
		// if(pointHessiansOut[i] && !pointHessiansOut[i]->Mp.expired())
		// 	pointHessiansOut[i]->Mp.lock()->setDirStatus(MapPoint::removed);
		delete pointHessiansOut[i];
	}
	for(unsigned int i=0;i<immaturePoints.size();i++) delete immaturePoints[i];

	pointHessians.clear();
	pointHessiansMarginalized.clear();
	pointHessiansOut.clear();
	immaturePoints.clear();
}


/**
 * @brief Make images
 * 
 * @param color 
 * @param HCalib 
 */
void FrameHessian::makeImages(float* color, CalibHessian* HCalib)
{

	// dIp contains (color, dx, dy)
	// absSquaredGrad contains the pixel gradient
	for(int i=0;i<pyrLevelsUsed;i++)
	{
		dIp[i] = new Eigen::Vector3f[wG[i]*hG[i]];
		std::fill(dIp[i], dIp[i]+wG[i]*hG[i], Eigen::Vector3f(0,0,0));
		absSquaredGrad[i] = new float[wG[i]*hG[i]];
		std::fill(absSquaredGrad[i], absSquaredGrad[i]+wG[i]*hG[i],0);
	}
	dI = dIp[0];


	// make d0
	int w=wG[0];
	int h=hG[0];
	for(int i=0;i<w*h;i++)
		dI[i][0] = color[i];

	for(int lvl=0; lvl<pyrLevelsUsed; lvl++)
	{
		int wl = wG[lvl], hl = hG[lvl];

		// Set dI_l and dabs_l to point to the dIp and absSquaredGrad arrays
		Eigen::Vector3f* dI_l = dIp[lvl];
		float* dabs_l = absSquaredGrad[lvl];
		if(lvl>0)
		{
			int lvlm1 = lvl-1;
			int wlm1 = wG[lvlm1];
			Eigen::Vector3f* dI_lm = dIp[lvlm1];

			// Take every other pixel from the last pyramid level
			// Smoothed by taking the average of the four pixels
			for(int y=0;y<hl;y++)
				for(int x=0;x<wl;x++)
				{
					dI_l[x + y*wl][0] = 0.25f * (dI_lm[2*x   + 2*y*wlm1][0] +
												dI_lm[2*x+1 + 2*y*wlm1][0] +
												dI_lm[2*x   + 2*y*wlm1+wlm1][0] +
												dI_lm[2*x+1 + 2*y*wlm1+wlm1][0]);
				}
		}

		for(int idx=0;idx < wl*hl;idx++)
		{
			// Derivative is appying a [1/2 0 -1/2] kernel in the x or y directions
			// Abosolute max value for derivative is 127.5
			// Derivative is calculated at the sides using extend 
			float dx = 0;
			if((idx%wl!=0) && (idx%wl!=(wl-1))) dx = 0.5f*(dI_l[idx+1][0] - dI_l[idx-1][0]);
			else if(idx%wl==0) dx = 0.5f*(dI_l[idx+1][0] - dI_l[idx][0]);
			else if(idx%wl==(wl-1)) dx = 0.5f*(dI_l[idx][0] - dI_l[idx-1][0]);

			float dy = 0;
			if((idx>=wl) && (idx<(wl*hl-wl))) dy = 0.5f*(dI_l[idx+wl][0] - dI_l[idx-wl][0]);
			else if(idx<wl) dy = 0.5f*(dI_l[idx+wl][0] - dI_l[idx][0]);
			else if(idx>=(wl*hl-wl)) dy = 0.5f*(dI_l[idx][0] - dI_l[idx-wl][0]);

			if(!std::isfinite(dx)) dx=0;
			if(!std::isfinite(dy)) dy=0;

			dI_l[idx][1] = dx;
			dI_l[idx][2] = dy;


			// Absolute max value for gradient is 32512.5
			dabs_l[idx] = dx*dx+dy*dy;

			if(setting_gammaWeightsPixelSelect==1 && HCalib!=0)
			{
				float gw = HCalib->getBGradOnly((float)(dI_l[idx][0]));
				// convert to gradient of original color space (before removing response) by correcting gamma
				dabs_l[idx] *= gw*gw;
			}
			// Normalize to 0-1
			dabs_l[idx] = dabs_l[idx]/32512.5;
		}
	}
}

void FrameHessian::makeColourImages(float* r, float* g ,float* b)
{
	colourValid = true;
	dI_c = new Eigen::Vector3f[wG[0]*hG[0]];
	std::fill(dI_c, dI_c+wG[0]*hG[0], Eigen::Vector3f(0,0,0));

	int w=wG[0];
	int h=hG[0];
	for(int i=0;i<w*h;i++){
		dI_c[i][0] = r[i];
		dI_c[i][1] = g[i];
		dI_c[i][2] = b[i];
	}
}

/**
 * @brief Set values that are commonly used in the optimization
 * 
 * Transformation matrixes, calibration matrix, and photogrammetric values
 * 
 * @param host 
 * @param target 
 * @param HCalib 
 */
void FrameFramePrecalc::set(FrameHessian* host, FrameHessian* target, CalibHessian* HCalib )
{
	this->host = host;
	this->target = target;

	// Transformation matrixes from host to target
	SE3 leftToLeft_0 = target->get_worldToCam_evalPT() * host->get_worldToCam_evalPT().inverse();
	PRE_RTll_0 = (leftToLeft_0.rotationMatrix()).cast<float>();
	PRE_tTll_0 = (leftToLeft_0.translation()).cast<float>();



	SE3 leftToLeft = target->PRE_worldToCam * host->PRE_camToWorld;
	PRE_RTll = (leftToLeft.rotationMatrix()).cast<float>();
	PRE_tTll = (leftToLeft.translation()).cast<float>();
	distanceLL = leftToLeft.translation().norm();


	// Calibration matrix K
	Mat33f K = Mat33f::Zero();
	K(0,0) = HCalib->fxl();
	K(1,1) = HCalib->fyl();
	K(0,2) = HCalib->cxl();
	K(1,2) = HCalib->cyl();
	K(2,2) = 1;
	// Pre-calculating values for effciency
	PRE_KRKiTll = K * PRE_RTll * K.inverse();
	PRE_RKiTll = PRE_RTll * K.inverse();
	PRE_KtTll = K * PRE_tTll;


	// Photogrammetric values
	PRE_aff_mode = AffLight::fromToVecExposure(host->ab_exposure, target->ab_exposure, host->aff_g2l(), target->aff_g2l()).cast<float>();
	PRE_b0_mode = host->aff_g2l_0().b;
}

}

