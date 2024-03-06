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

#include "FullSystem/ImmaturePoint.h"
#include "util/FrameShell.h"
#include "FullSystem/ResidualProjections.h"

namespace HSLAM
{

/**
 * @brief Construct a new immature point object
 * 
 * @param u_ 		x position of point
 * @param v_ 		y position of point
 * @param host_ 	Frame containing point
 * @param type 		Pyramid level point is detected in
 * @param HCalib 
 */
ImmaturePoint::ImmaturePoint(int u_, int v_, FrameHessian* host_, float type, CalibHessian* HCalib)
: u(u_), v(v_), host(host_), my_type(type), idepth_min(0), idepth_max(NAN), lastTraceStatus(IPS_UNINITIALIZED)
{

	gradH.setZero();

	colourValid = false;
	if(host->colourValid) colourValid = true;

	// Add all pixels in the pattern to the point
	for(int idx=0;idx<PATTERNNUM;idx++)
	{
		int dx = PATTERNP[idx][0];
		int dy = PATTERNP[idx][1];

		// ptc is (pixel intensity, dx, dy)
		Vec3f ptc = getInterpolatedElement33BiLin(host->dI, u+dx, v+dy,wG[0]);
		color[idx] = ptc[0]; // Set pixel internsity

		if(colourValid){
			int u_int = static_cast<int>(u);
			int v_int = static_cast<int>(v);
			Vec3f ptc_3 = host->dI_c[u_int+dx+(v_int+dy)*wG[0]];
			colour3[idx] = ptc_3;
		}

		if(!std::isfinite(color[idx])) {energyTH=NAN; return;}


		gradH += ptc.tail<2>() * ptc.tail<2>().transpose(); // 2 by 2 matrix of summed dx^2 + dy^2 values
		// Weights for gradient-dependent weighting

		weights[idx] = sqrtf(setting_outlierTHSumComponent / (setting_outlierTHSumComponent + ptc.tail<2>().squaredNorm()));
	}

	energyTH = PATTERNNUM*setting_outlierTH;
	energyTH *= setting_overallEnergyTHWeight*setting_overallEnergyTHWeight;

	idepth_GT=0;
	quality=10000;

}

/**
 * @brief Destroy the Immature object
 * 
 */
ImmaturePoint::~ImmaturePoint()
{
}



/**
 * @brief Traces the points as the frames progress
 * 
 * Steps:
 * 1. Project (u,v) point assuming min and max depth
 * 2. Search along line between the max and min depth points uniformly for lowest energy
 * 3. Do Gauss Newton optimization with found point to get a more precise point
 * 
 * @param frame 				// Frame that point in contained in
 * @param hostToFrame_KRKi		// Rotation matrix from frame movement
 * @param hostToFrame_Kt 		// Translation matrix from frame movement
 * @param hostToFrame_affine	// Constants for photogrammetric conversion
 * @param HCalib 
 * @param debugPrint 
 * @return ImmaturePointStatus: 
 * OOB -> point is optimized and marginalized.
 * UPDATED -> point has been updated.
 * SKIP -> point has not been updated.
 */
ImmaturePointStatus ImmaturePoint::traceOn(FrameHessian* frame,const Mat33f &hostToFrame_KRKi, const Vec3f &hostToFrame_Kt, const Vec2f& hostToFrame_affine, CalibHessian* HCalib, bool debugPrint)
{
	if(lastTraceStatus == ImmaturePointStatus::IPS_OOB) return lastTraceStatus;


	debugPrint = false;
	float maxPixSearch = (wG[0]+hG[0])*setting_maxPixSearch;

	if(debugPrint)
		printf("trace pt (%.1f %.1f) from frame %d to %d. Range %f -> %f. t %f %f %f!\n",
				u,v,
				host->shell->id, frame->shell->id,
				idepth_min, idepth_max,
				hostToFrame_Kt[0],hostToFrame_Kt[1],hostToFrame_Kt[2]);

//	const float stepsize = 1.0;				// stepsize for initial discrete search.
//	const int GNIterations = 3;				// max # GN iterations
//	const float GNThreshold = 0.1;				// GN stop after this stepsize.
//	const float extraSlackOnTH = 1.2;			// for energy-based outlier check, be slightly more relaxed by this factor.
//	const float slackInterval = 0.8;			// if pixel-interval is smaller than this, leave it be.
//	const float minImprovementFactor = 2;		// if pixel-interval is smaller than this, leave it be.
	
	// ============== Project assumming min and max depth. Return if one of them is OOB ===================
	Vec3f pr = hostToFrame_KRKi * Vec3f(u,v,1); // Unproject (u,v) (depth does not matter for rotation), rotate, and the re-project
	
	// Translation is dependent on depth, which is unknown, so depth must be estimated
	Vec3f ptpMin = pr + hostToFrame_Kt*idepth_min;	// Translate rotated (u,v) point assuming depth is idepth_min
	float uMin = ptpMin[0] / ptpMin[2];
	float vMin = ptpMin[1] / ptpMin[2];

	// Find the maximum size of the pattern
	Mat22f Rplane = hostToFrame_KRKi.topLeftCorner<2,2>();
	int maxRotPatX = 0;
	int maxRotPatY = 0;
	Vec2f rotatetPattern[MAX_RES_PER_POINT];
	for(int idx=0;idx<PATTERNNUM;idx++)
	{
	rotatetPattern[idx] = Rplane * Vec2f(PATTERNP[idx][0], PATTERNP[idx][1]);
	int absX = (int) abs(rotatetPattern[idx][0]);
	int absY = (int) abs(rotatetPattern[idx][1]);
	maxRotPatX = std::max(absX, maxRotPatX);
	maxRotPatY = std::max(absY, maxRotPatY);
	}
	int realBoundU = maxRotPatX + 2;
	int realBoundV = maxRotPatY + 2;
	int boundU = 4;
	int boundV = 4;
	boundU = std::max(boundU, realBoundU);
	boundV = std::max(boundV, realBoundV);

	if(!(uMin > boundU && vMin > boundV && uMin < wG[0]-boundU-1 && vMin < hG[0]-boundV-1)) // Pattern is OOB
	{
		if(debugPrint) printf("OOB uMin %f %f - %f %f %f (id %f-%f)!\n",
				u,v,uMin, vMin,  ptpMin[2], idepth_min, idepth_max);
		lastTraceUV = Vec2f(-1,-1);
		lastTracePixelInterval=0;
		return lastTraceStatus = ImmaturePointStatus::IPS_OOB;
	}

	float dist;
	float uMax;
	float vMax;
	Vec3f ptpMax;
	if(std::isfinite(idepth_max))
	{
		ptpMax = pr + hostToFrame_Kt*idepth_max; // Repeat, but assuming the depth is idepth_max
		uMax = ptpMax[0] / ptpMax[2];
		vMax = ptpMax[1] / ptpMax[2];


		if(!(uMax > boundU && vMax > boundV && uMax < wG[0]-boundU-1 && vMax < hG[0]-boundV-1)) // Pattern is OOB
		{
			if(debugPrint) printf("OOB uMax  %f %f - %f %f!\n",u,v, uMax, vMax);
			lastTraceUV = Vec2f(-1,-1);
			lastTracePixelInterval=0;
			return lastTraceStatus = ImmaturePointStatus::IPS_OOB;
		}



		// ============== check their distance. everything below 2px is OK (-> skip). ===================
		dist = (uMin-uMax)*(uMin-uMax) + (vMin-vMax)*(vMin-vMax);
		dist = sqrtf(dist);
		if(dist < setting_trace_slackInterval) // Tracing is not needed if the distance between max and min is low
		{
			if(debugPrint)
				printf("TOO CERTAIN ALREADY (dist %f)!\n", dist);

			lastTraceUV = Vec2f(uMax+uMin, vMax+vMin)*0.5; // Take average
			lastTracePixelInterval=dist;
			return lastTraceStatus = ImmaturePointStatus::IPS_SKIPPED;
		}
		assert(dist>0);
	}
	else // Intial search will start off with max depth at infinite
	{
		dist = maxPixSearch; // Set length of search line to maxPixSearch

		// Project to arbitrary depth to get direction
		ptpMax = pr + hostToFrame_Kt*0.01;
		uMax = ptpMax[0] / ptpMax[2];
		vMax = ptpMax[1] / ptpMax[2];
		// Direction.
		float dx = uMax-uMin;
		float dy = vMax-vMin;
		float d = 1.0f / sqrtf(dx*dx+dy*dy);

		// Set to [setting_maxPixSearch].
		uMax = uMin + dist*dx*d;
		vMax = vMin + dist*dy*d;

		if(!(uMax > boundU && vMax > boundV && uMax < wG[0]-boundU-1 && vMax < hG[0]-boundV-1)) // Pattern may still be OOB
		{
			if(debugPrint) printf("OOB uMax-coarse %f %f %f!\n", uMax, vMax,  ptpMax[2]);
			lastTraceUV = Vec2f(-1,-1);
			lastTracePixelInterval=0;
			return lastTraceStatus = ImmaturePointStatus::IPS_OOB;
		}
		assert(dist>0);
	}


	if(!(idepth_min<0 || (ptpMin[2]>0.75 && ptpMin[2]<1.5))) // set OOB if scale is too big
	{
		if(debugPrint) printf("OOB SCALE %f %f %f!\n", uMax, vMax,  ptpMin[2]);
		lastTraceUV = Vec2f(-1,-1);
		lastTracePixelInterval=0;
		return lastTraceStatus = ImmaturePointStatus::IPS_OOB;
	}


	// ============== compute error-bounds on result in pixel. if the new interval is not at least 1/2 of the old, SKIP ===================
	float dx = setting_trace_stepsize*(uMax-uMin);
	float dy = setting_trace_stepsize*(vMax-vMin);

	float a_err = (Vec2f(dx,dy).transpose() * gradH * Vec2f(dx,dy));
	float b_err = (Vec2f(dy,-dx).transpose() * gradH * Vec2f(dy,-dx)); // Perpendicular to a going clockwise
	float errorInPixel = 0.2f + 0.2f * (a_err+b_err) / a_err;

	if(errorInPixel*setting_trace_minImprovementFactor > dist && std::isfinite(idepth_max))
	{
		if(debugPrint)
			printf("NO SIGNIFICANT IMPROVMENT (%f)!\n", errorInPixel);
		lastTraceUV = Vec2f(uMax+uMin, vMax+vMin)*0.5;
		lastTracePixelInterval=dist;
		return lastTraceStatus = ImmaturePointStatus::IPS_BADCONDITION;
	}

	if(errorInPixel >10) errorInPixel=10;



	// ============== do the discrete search ===================
	dx /= dist;
	dy /= dist;

	if(debugPrint)
		printf("trace pt (%.1f %.1f) from frame %d to %d. Range %f (%.1f %.1f) -> %f (%.1f %.1f)! ErrorInPixel %.1f!\n",
				u,v,
				host->shell->id, frame->shell->id,
				idepth_min, uMin, vMin,
				idepth_max, uMax, vMax,
				errorInPixel
				);


	if(dist>maxPixSearch)
	{
		uMax = uMin + maxPixSearch*dx;
		vMax = vMin + maxPixSearch*dy;
		dist = maxPixSearch;
	}

	int numSteps = 1.9999f + dist / setting_trace_stepsize;
	// Mat22f Rplane = hostToFrame_KRKi.topLeftCorner<2,2>();

	// Add some randomness
	float randShift = uMin*1000-floorf(uMin*1000);
	float ptx = uMin-randShift*dx;
	float pty = vMin-randShift*dy;

	/*
	Vec2f rotatetPattern[MAX_RES_PER_POINT];
	for(int idx=0;idx<PATTERNNUM;idx++)
		rotatetPattern[idx] = Rplane * Vec2f(PATTERNP[idx][0], PATTERNP[idx][1]);
	*/

	if(!std::isfinite(dx) || !std::isfinite(dy)) // OOD if dx or dy is infinite
	{
		printf("CAUGHT INF / NAN dxdy!\n");

		lastTracePixelInterval=0;
		lastTraceUV = Vec2f(-1,-1);
		return lastTraceStatus = ImmaturePointStatus::IPS_OOB;
	}



	// Start searching starting from the uMin,vMin values
	// Get point between the min and max depth points with the lowest energy
	float errors[100];
	float bestU=0, bestV=0, bestEnergy=1e10;
	int bestIdx=-1;
	if(numSteps >= 100) numSteps = 99;

	for(int i=0;i<numSteps;i++)
	{
		float energy=0;
		// Calculate residual of pattern at each searched point
		for(int idx=0;idx<PATTERNNUM;idx++)
		{
			float hitColor = getInterpolatedElement31(frame->dI,
										(float)(ptx+rotatetPattern[idx][0]),
										(float)(pty+rotatetPattern[idx][1]),
										wG[0]);

			if(!std::isfinite(hitColor)) {energy+=1e5; continue;}
			float residual = hitColor - (float)(hostToFrame_affine[0] * color[idx] + hostToFrame_affine[1]);
			float hw = fabs(residual) < setting_huberTH ? 1 : setting_huberTH / fabs(residual);
			energy += hw *residual*residual*(2-hw);
		}

		if(debugPrint)
			printf("step %.1f %.1f (id %f): energy = %f!\n",
					ptx, pty, 0.0f, energy);


		errors[i] = energy;
		if(energy < bestEnergy)
		{
			bestU = ptx; bestV = pty; bestEnergy = energy; bestIdx = i;
		}

		ptx+=dx;
		pty+=dy;
	}


	// Find best score outside a +-2px radius.
	float secondBest=1e10;
	for(int i=0;i<numSteps;i++)
	{
		if((i < bestIdx-setting_minTraceTestRadius || i > bestIdx+setting_minTraceTestRadius) && errors[i] < secondBest)
			secondBest = errors[i];
	}
	float newQuality = secondBest / bestEnergy;
	if(newQuality < quality || numSteps > 10) quality = newQuality;


	// ============== do Gauss Newton optimization ===================
	float uBak=bestU, vBak=bestV, gnstepsize=1, stepBack=0;
	if(setting_trace_GNIterations>0) bestEnergy = 1e5;
	int gnStepsGood=0, gnStepsBad=0;
	for(int it=0;it<setting_trace_GNIterations;it++)
	{
		float H = 1, b=0, energy=0;
		for(int idx=0;idx<PATTERNNUM;idx++)
		{
			float posU = (float)(bestU + rotatetPattern[idx][0]);
			float posV = (float)(bestV + rotatetPattern[idx][1]);

			if(posU < 0 || posV < 0 || posU >= wG[0] - 1 || posV >= hG[0] - 1) // Check if pattern is OOB
			{
				if(debugPrint) printf("OOB uMax  %f %f - %f %f!\n", posU, posV, uMax, vMax);
				lastTraceUV = Vec2f(-1,-1);
				lastTracePixelInterval=0;
				return lastTraceStatus = ImmaturePointStatus::IPS_OOB;
			}

			Vec3f hitColor = getInterpolatedElement33(frame->dI, posU, posV, wG[0]);

			if(!std::isfinite((float)hitColor[0])) {energy+=1e5; continue;}
			float residual = hitColor[0] - (hostToFrame_affine[0] * color[idx] + hostToFrame_affine[1]);
			float dResdDist = dx*hitColor[1] + dy*hitColor[2]; // Calculate derivative for GN
			float hw = fabs(residual) < setting_huberTH ? 1 : setting_huberTH / fabs(residual);

			H += hw*dResdDist*dResdDist;
			b += hw*residual*dResdDist;
			energy += weights[idx]*weights[idx]*hw *residual*residual*(2-hw);
		}


		if(energy > bestEnergy)
		{
			gnStepsBad++;

			// Do a smaller step from old point.
			stepBack*=0.5;
			bestU = uBak + stepBack*dx;
			bestV = vBak + stepBack*dy;
			if(debugPrint)
				printf("GN BACK %d: E %f, H %f, b %f. id-step %f. UV %f %f -> %f %f.\n",
						it, energy, H, b, stepBack,
						uBak, vBak, bestU, bestV);
		}
		else
		{
			// Step towards to optimal point
			gnStepsGood++;

			float step = -gnstepsize*b/H;
			if(step < -0.5) step = -0.5;
			else if(step > 0.5) step=0.5;

			if(!std::isfinite(step)) step=0;

			uBak=bestU;
			vBak=bestV;
			stepBack=step;

			bestU += step*dx;
			bestV += step*dy;
			bestEnergy = energy;

			if(debugPrint)
				printf("GN step %d: E %f, H %f, b %f. id-step %f. UV %f %f -> %f %f.\n",
						it, energy, H, b, step,
						uBak, vBak, bestU, bestV);
		}

		if(fabsf(stepBack) < setting_trace_GNThreshold) break;
	}


	// ============== detect energy-based outlier. ===================
//	float absGrad0 = getInterpolatedElement(frame->absSquaredGrad[0],bestU, bestV, wG[0]);
//	float absGrad1 = getInterpolatedElement(frame->absSquaredGrad[1],bestU*0.5-0.25, bestV*0.5-0.25, wG[1]);
//	float absGrad2 = getInterpolatedElement(frame->absSquaredGrad[2],bestU*0.25-0.375, bestV*0.25-0.375, wG[2]);
	if(!(bestEnergy < energyTH*setting_trace_extraSlackOnTH))
//			|| (absGrad0*areaGradientSlackFactor < host->frameGradTH
//		     && absGrad1*areaGradientSlackFactor < host->frameGradTH*0.75f
//			 && absGrad2*areaGradientSlackFactor < host->frameGradTH*0.50f))
	{
		if(debugPrint)
			printf("OUTLIER!\n");

		lastTracePixelInterval=0;
		lastTraceUV = Vec2f(-1,-1);
		if(lastTraceStatus == ImmaturePointStatus::IPS_OUTLIER)
			return lastTraceStatus = ImmaturePointStatus::IPS_OOB;
		else
			return lastTraceStatus = ImmaturePointStatus::IPS_OUTLIER;
	}


	// ============== set new interval for idepth_min and idepth_max===================
	if(dx*dx>dy*dy)
	{
		idepth_min = (pr[2]*(bestU-errorInPixel*dx) - pr[0]) / (hostToFrame_Kt[0] - hostToFrame_Kt[2]*(bestU-errorInPixel*dx));
		idepth_max = (pr[2]*(bestU+errorInPixel*dx) - pr[0]) / (hostToFrame_Kt[0] - hostToFrame_Kt[2]*(bestU+errorInPixel*dx));
	}
	else
	{
		idepth_min = (pr[2]*(bestV-errorInPixel*dy) - pr[1]) / (hostToFrame_Kt[1] - hostToFrame_Kt[2]*(bestV-errorInPixel*dy));
		idepth_max = (pr[2]*(bestV+errorInPixel*dy) - pr[1]) / (hostToFrame_Kt[1] - hostToFrame_Kt[2]*(bestV+errorInPixel*dy));
	}
	if(idepth_min > idepth_max) std::swap<float>(idepth_min, idepth_max);


	if(!std::isfinite(idepth_min) || !std::isfinite(idepth_max) || (idepth_max<0))
	{
		if(!setting_debugout_runquiet) printf("CAUGHT INF / NAN minmax depth!\n");

		lastTracePixelInterval=0;
		lastTraceUV = Vec2f(-1,-1);
		return lastTraceStatus = ImmaturePointStatus::IPS_OUTLIER;
	}

	lastTracePixelInterval=2*errorInPixel;
	lastTraceUV = Vec2f(bestU, bestV);
	return lastTraceStatus = ImmaturePointStatus::IPS_GOOD;
}


float ImmaturePoint::getdPixdd(
		CalibHessian *  HCalib,
		ImmaturePointTemporaryResidual* tmpRes,
		float idepth)
{
	FrameFramePrecalc* precalc = &(host->targetPrecalc[tmpRes->target->idx]);
	const Vec3f &PRE_tTll = precalc->PRE_tTll;
	float drescale, u=0, v=0, new_idepth;
	float Ku, Kv;
	Vec3f KliP;

	projectPoint(this->u,this->v, idepth, 0, 0,HCalib,
			precalc->PRE_RTll,PRE_tTll, drescale, u, v, Ku, Kv, KliP, new_idepth);

	float dxdd = (PRE_tTll[0]-PRE_tTll[2]*u)*HCalib->fxl();
	float dydd = (PRE_tTll[1]-PRE_tTll[2]*v)*HCalib->fyl();
	return drescale*sqrtf(dxdd*dxdd + dydd*dydd);
}


float ImmaturePoint::calcResidual(
		CalibHessian *  HCalib, const float outlierTHSlack,
		ImmaturePointTemporaryResidual* tmpRes,
		float idepth)
{
	FrameFramePrecalc* precalc = &(host->targetPrecalc[tmpRes->target->idx]);

	float energyLeft=0;
	const Eigen::Vector3f* dIl = tmpRes->target->dI;
	const Mat33f &PRE_KRKiTll = precalc->PRE_KRKiTll;
	const Vec3f &PRE_KtTll = precalc->PRE_KtTll;
	Vec2f affLL = precalc->PRE_aff_mode;

	for(int idx=0;idx<PATTERNNUM;idx++)
	{
		float Ku, Kv;
		if(!projectPoint(this->u+PATTERNP[idx][0], this->v+PATTERNP[idx][1], idepth, PRE_KRKiTll, PRE_KtTll, Ku, Kv))
			{return 1e10;}

		Vec3f hitColor = (getInterpolatedElement33(dIl, Ku, Kv, wG[0]));
		if(!std::isfinite((float)hitColor[0])) {return 1e10;}

		float residual = hitColor[0] - (affLL[0] * color[idx] + affLL[1]);

		float hw = fabsf(residual) < setting_huberTH ? 1 : setting_huberTH / fabsf(residual);
		energyLeft += weights[idx]*weights[idx]*hw *residual*residual*(2-hw);
	}

	if(energyLeft > energyTH*outlierTHSlack)
	{
		energyLeft = energyTH*outlierTHSlack;
	}
	return energyLeft;
}


/**
 * @brief Calculate some residual values for the immature points
 * 
 * Will not calculate the Hessian
 * Calculates the energy residual and values that are helpful for point optimizations
 * 
 * @param HCalib 			Calibration matrices
 * @param outlierTHSlack 	Outlier threshold
 * @param tmpRes 			Struct storing residual information
 * @param Hdd 				
 * @param bd 				
 * @param idepth 			Input inverse depth
 * @return double 			EnergyLeft
 */
double ImmaturePoint::linearizeResidual(
		CalibHessian *  HCalib, const float outlierTHSlack,
		ImmaturePointTemporaryResidual* tmpRes,
		float &Hdd, float &bd,
		float idepth)
{
	if(tmpRes->state_state == ResState::OOB)
		{ tmpRes->state_NewState = ResState::OOB; return tmpRes->state_energy; }

	FrameFramePrecalc* precalc = &(host->targetPrecalc[tmpRes->target->idx]);

	float energyLeft=0;
	// Get image
	const Eigen::Vector3f* dIl = tmpRes->target->dI;
	// Set transforms
	const Mat33f &PRE_RTll = precalc->PRE_RTll;
	const Vec3f &PRE_tTll = precalc->PRE_tTll;
	//const float * const Il = tmpRes->target->I;

	// Get photometric values
	Vec2f affLL = precalc->PRE_aff_mode;

	for(int idx=0;idx<PATTERNNUM;idx++)
	{
		int dx = PATTERNP[idx][0];
		int dy = PATTERNP[idx][1];

		float drescale, u, v, new_idepth;
		float Ku, Kv;
		Vec3f KliP;

		// Get intensity of point at transformed location
		if(!projectPoint(this->u,this->v, idepth, dx, dy,HCalib,
				PRE_RTll,PRE_tTll, drescale, u, v, Ku, Kv, KliP, new_idepth))
			{tmpRes->state_NewState = ResState::OOB; return tmpRes->state_energy;}


		Vec3f hitColor = (getInterpolatedElement33(dIl, Ku, Kv, wG[0]));

		if(!std::isfinite((float)hitColor[0])) {tmpRes->state_NewState = ResState::OOB; return tmpRes->state_energy;}

		// Calculate residual
		float residual = hitColor[0] - (affLL[0] * color[idx] + affLL[1]);
		// Huber loss
		float hw = fabsf(residual) < setting_huberTH ? 1 : setting_huberTH / fabsf(residual);
		energyLeft += weights[idx]*weights[idx]*hw *residual*residual*(2-hw);

		// depth derivatives.
		float dxInterp = hitColor[1]*HCalib->fxl();
		float dyInterp = hitColor[2]*HCalib->fyl();
		float d_idepth = derive_idepth(PRE_tTll, u, v, dx, dy, dxInterp, dyInterp, drescale);

		// Huber weight
		hw *= weights[idx]*weights[idx];

		Hdd += (hw*d_idepth)*d_idepth;
		bd += (hw*residual)*d_idepth;
	}


	if(energyLeft > energyTH*outlierTHSlack)
	{
		energyLeft = energyTH*outlierTHSlack;
		tmpRes->state_NewState = ResState::OUTLIER;
	}
	else
	{
		tmpRes->state_NewState = ResState::IN;
	}

	tmpRes->state_NewEnergy = energyLeft;
	return energyLeft;
}
}
