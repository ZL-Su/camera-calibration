#include <iostream>
#include <mono_calibrator.h>

using calibration_type = dgelom::mono_calibrator<double>;

int main() {
	calibration_type::image_info_t fname("E:\\SciR\\Datum\\calib_imgs");

	return 0;
}