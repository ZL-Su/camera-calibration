#include <iostream>
#include <mono_calibrator.h>

using calibration_type = dgelom::mono_calibrator<double_t>;

int main() {
	calibration_type::image_info_t fname("E:\\SciR\\Datum\\calib_imgs\\7x7");

	calibration_type driver(fname, { 7, 7, 26.f });

	decltype(auto) params = driver.run();
	std::cout << " >> Internal parameters:";
	dgelom::print(params);

	decltype(auto) poses = driver(0);
	std::cout << " >> External parameters:";
	dgelom::print(poses);

	driver.image_points() = driver.image_points();
	params = driver.run(false);
	dgelom::print(params);

	std::cout << " Reprojection error: " << driver.error() << "\n";

return 0;
}