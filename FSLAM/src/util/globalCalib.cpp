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


#include "util/globalCalib.h"
#include "stdio.h"
#include <iostream>

namespace HSLAM
{
	// Image width and height
	int wG[PYR_LEVELS], hG[PYR_LEVELS];
	// Values of K matrix
	float fxG[PYR_LEVELS], fyG[PYR_LEVELS],
		  cxG[PYR_LEVELS], cyG[PYR_LEVELS];

	// Values of inverse K matrix
	float fxiG[PYR_LEVELS], fyiG[PYR_LEVELS],
		  cxiG[PYR_LEVELS], cyiG[PYR_LEVELS];

	// K and K inverse matrix
	Eigen::Matrix3f KG[PYR_LEVELS], KiG[PYR_LEVELS];


	// Image width and height at first level accounting for borders
	float wM3G;
	float hM3G;

	void setGlobalCalib(int w, int h,const Eigen::Matrix3f &K)
	{
		int wlvl=w;
		int hlvl=h;
		pyrLevelsUsed=1;
		// Determine number of image pyramid levels to use
		while(wlvl%2==0 && hlvl%2==0 && wlvl*hlvl > 5000 && pyrLevelsUsed < PYR_LEVELS)
		{
			wlvl /=2;
			hlvl /=2;
			pyrLevelsUsed++;
		}
		printf("using pyramid levels 0 to %d. coarsest resolution: %d x %d!\n",
				pyrLevelsUsed-1, wlvl, hlvl);
		if(wlvl>100 && hlvl > 100)
		{
			printf("\n\n===============WARNING!===================\n "
					"using not enough pyramid levels.\n"
					"Consider scaling to a resolution that is a multiple of a power of 2.\n");
		}
		if(pyrLevelsUsed < 3)
		{
			printf("\n\n===============WARNING!===================\n "
					"I need higher resolution.\n"
					"I will probably segfault.\n");
		}

		// Set values for each pyramid level
		wM3G = w-3;
		hM3G = h-3;

		wG[0] = w;
		hG[0] = h;
		KG[0] = K;
		fxG[0] = K(0,0);
		fyG[0] = K(1,1);
		cxG[0] = K(0,2);
		cyG[0] = K(1,2);
		KiG[0] = KG[0].inverse();
		fxiG[0] = KiG[0](0,0);
		fyiG[0] = KiG[0](1,1);
		cxiG[0] = KiG[0](0,2);
		cyiG[0] = KiG[0](1,2);

		for (int level = 1; level < pyrLevelsUsed; ++ level)
		{
			wG[level] = w >> level;
			hG[level] = h >> level;

			fxG[level] = fxG[level-1] * 0.5;
			fyG[level] = fyG[level-1] * 0.5;
			cxG[level] = (cxG[0] + 0.5) / ((int)1<<level) - 0.5;
			cyG[level] = (cyG[0] + 0.5) / ((int)1<<level) - 0.5;

			KG[level] << fxG[level], 0.0, cxG[level], 
						 0.0, fyG[level], cyG[level], 
						 0.0, 0.0, 1.0;	// synthetic
			KiG[level] = KG[level].inverse();

			fxiG[level] = KiG[level](0,0);
			fyiG[level] = KiG[level](1,1);
			cxiG[level] = KiG[level](0,2);
			cyiG[level] = KiG[level](1,2);
		}
	}


}
