#include <iostream>
#include <mono_calibrator.h>

using calibration_type = dgelom::mono_calibrator<double>;

int main() {
	calibration_type::image_info_t fname("E:\\SciR\\Datum\\calib_imgs");

	calibration_type driver(fname, { 8, 11, 5.f });

	decltype(auto) internal_pars = driver.run();

	decltype(auto) valid_indices = driver.valid_image_indices();
	for (const auto idx : valid_indices) {
		auto r_and_t = driver(idx);
	}

	return 0;
}