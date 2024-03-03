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

// Because histogram is only used to calculate quartiles
// Number of bins only affects quartile resolution
// Fixed to 100 to get a resolution of 1%
#define NUM_BINS 100

#define DEBUG_MESSAGE_PIXELSELECTOR2 0



namespace HSLAM
{

enum PixelSelectorStatus {PIXSEL_VOID=0, PIXSEL_1, PIXSEL_2, PIXSEL_3};


class FrameHessian;

class PixelSelector
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
	int makeMaps( const FrameHessian* const fh, float* map_out, float density, int recursionsLeft=1, float thFactor=1);

	PixelSelector(int w, int h);
	~PixelSelector();

	void makeThresTable(const FrameHessian* const fh);

	int currentPotential;

	bool debugPlot;
private:
	Eigen::Vector3i select(const FrameHessian* const fh,
			float* map_out, int pot, float thFactor=1);


	unsigned char* randomPattern;


	// Table of minimum gradient threshold
	float* ths;
	float* thsSmoothed;
	int thsStep;
	const FrameHessian* gradHistFrame;

	// block width, and block height.
	int bW, bH;
	// number of blocks in x and y dimension.
	int nbW, nbH;
};

}

