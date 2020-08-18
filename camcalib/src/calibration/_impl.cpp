#include "../cost/cost_functions.hpp"
#include "../../include/calibration/structure_self_calibration.h"

namespace dgelom {
struct SolverConfig
{
	using options_t = xop::Solver::Options;
	static inline options_t get_options(double tol = 1.0e-8)
	{
		options_t _Ret;
		_Ret.linear_solver_type = xop::SPARSE_SCHUR;
		_Ret.sparse_linear_algebra_library_type = xop::SUITE_SPARSE;
		_Ret.minimizer_type = xop::TRUST_REGION;
		_Ret.trust_region_strategy_type = xop::LEVENBERG_MARQUARDT;
		_Ret.preconditioner_type = xop::CLUSTER_JACOBI;
		_Ret.visibility_clustering_type = xop::SINGLE_LINKAGE;
		_Ret.max_num_iterations = 100;
		_Ret.minimizer_progress_to_stdout = true;
		_Ret.function_tolerance = tol;
		_Ret.gradient_tolerance = tol;
		_Ret.parameter_tolerance = tol;
		_Ret.num_threads = 1;
		_Ret.num_linear_solver_threads = 1;
		return _Ret;
	}
};
}

template<typename T, size_t _DescRadius, size_t _PatchRadius>
void dgelom::internal::_Impl(const dge::Matrix<T>& _Map, std::vector<Feature_<T, _DescRadius, _PatchRadius>>& Feats)
{
	constexpr auto b = dgelom::max(2, _PatchRadius);
	const auto max_rows = _Map.rows() - b - 1;
	const auto max_cols = _Map.cols() - b - 1;
	is_local_maximum<std::remove_reference_t<decltype(_Map)>, _DescRadius> is_max(_Map);
	Feature_<T, _DescRadius, _PatchRadius> _F;
	for (int y = b; y < max_rows; ++y)
	{
		const auto row = _Map.ptr(y);
		for (int x = b; x < max_cols; ++x)
		{
			auto _Sali = row[x];
			if (_Sali >0 && is_max(x, y))
			{
				_F.point() = { T(x), T(y) };
				_F.saliency() = _Sali;
				Feats.emplace_back(_F);
			}
		}
	}
}

template<typename _Arg> void dgelom::internal::_Impl(_Arg * arg)
{
	auto& _Options = arg->options();
	auto& _Results = arg->results();
	auto& _Dataset = arg->dataset();
	auto& _Featurs = arg->features();
	auto& _K = _Results._K;
	auto& _T = _Results._T;
	auto& _W = _Results._Invdepth;

	xop::Problem problem;
	for (size_t k = 0; k < _Featurs.size(); ++k)
	{
		auto& _P = _Featurs[k];
		auto& _G = _Dataset._Gradts[1];
		using Fty = std::remove_reference_t<decltype(_P)>;
		using Gty = std::remove_reference_t<decltype(_G)>;
		auto cost = CostFunc::_Phm_error<Fty, Gty>::Create(_P, _G);
		problem.AddResidualBlock(cost, new xop::HuberLoss(0.05), _K[0], _K[1], _T.data(), _W[k]);
	}

	xop::Solver::Summary summary;
	auto options = SolverConfig::get_options();
	xop::Solve(options, &problem, &summary);

	std::cout << summary.FullReport() << std::endl;
	std::cout << "[MSG] Done!" << std::endl;
}

template<> void dgelom::internal::_Impl(dgelom::Calibration_<float, dgelom::Options<CON>>* arg)
{
	auto& _Options = arg->options();
	auto& _Results = arg->results();
	auto& _Dataset = arg->dataset();
	auto& _K = _Results._K;
	auto& _T = _Results._T;
	auto& _W = _Results._Invdepth;
	const auto& _Featurs = arg->m_features;

	const auto _Size = _Featurs.size() >> 1;
	xop::Problem problem;
	for (size_t k = 0; k < _Size; ++k)
	{
		const auto& _P0 = _Featurs[k];
		const auto& _P1 = _Featurs[k + _Size];
		using point_type = std::remove_reference_t<decltype(_P0.point())>;
		auto cost = CostFunc::_Std_error<point_type>::Create(_P0.point(), _P1.point());
		problem.AddResidualBlock(cost, new xop::HuberLoss(0.1), _K[0], _K[1], _T.data(), _W[k]);
	}

	xop::Solver::Summary summary;
	auto options = SolverConfig::get_options();
	xop::Solve(options, &problem, &summary);

	std::cout << summary.FullReport() << std::endl;
	std::cout << "[MSG] Done!" << std::endl;
}

template<> void dgelom::internal::_Impl(dgelom::stereo_calibrator<0>* _Arg)
{
	const auto& _Points_pair = _Arg->correspondences();
	const auto& _Left = _Points_pair.first;
	const auto& _Right = _Points_pair.second;

	auto& _Pose = _Arg->poses().back();
	auto& _Depth = _Arg->depths().back();

	auto _Uniform = std::bind(std::uniform_real_distribution<double_t>{1e-6, 5e-6}, std::default_random_engine());
	for_each(_Depth, [&](auto& _Val) {return _Val = _Uniform(); });

	auto _Problem = std::make_shared<xop::Problem>();

	size_t _Pos = 0;
	for (auto L = _Left.rwbegin(), R = _Right.rwbegin(); L; ++L, ++R) {
		auto _Cost = dgelom::CostFunc::_Lsq_stereo_errr<>::Create(L, R);
		_Problem->AddResidualBlock(_Cost, new xop::HuberLoss(1.0), _Pose.data(), _Depth[_Pos++]);
	}

	auto _Options = SolverConfig::get_options();

	auto _Summary = std::make_unique<xop::Solver::Summary>();
	xop::Solve(_Options, _Problem.get(), _Summary.get());
	std::cout << _Summary->FullReport() << std::endl;

	_Depth = 1.0 / _Depth;

}

template void dgelom::internal::_Impl(const dge::Matrix<float>&, std::vector<Feature_<float, 1, 2>>&);
template void dgelom::internal::_Impl(dgelom::Calibration_<float, dgelom::Options<UNC, 1, 2>>*);
template void dgelom::internal::_Impl(const dge::Matrix<float>&, std::vector<Feature_<float, 3, 2>>&);
template void dgelom::internal::_Impl(dgelom::Calibration_<float, dgelom::Options<UNC, 3, 2>>*);
